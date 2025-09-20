#!/usr/bin/env python3
"""
Graph Neural Networks for MPEG-G Microbiome Challenge

This module implements comprehensive GNN architectures to capture microbiome
interaction patterns, including co-occurrence networks, functional pathways,
and temporal dynamics for innovative cytokine prediction approaches.

Features:
- Multiple network construction methods (correlation, co-occurrence, functional)
- Various GNN architectures (GCN, GAT, GraphSAGE, Graph Transformer)
- Temporal graph analysis for T1/T2 dynamics
- Biological network integration
- Advanced graph pooling and multi-scale representations
- Interpretability analysis and visualization
"""

import os
import sys
import json
import logging
import warnings
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import networkx as nx

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, TransformerConv,
    global_mean_pool, global_max_pool, global_add_pool,
    TopKPooling, SAGPooling, ASAPooling
)
from torch_geometric.utils import to_networkx, from_networkx

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gnn_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MicrobiomeNetworkConstructor:
    """Constructs various types of microbiome networks for GNN analysis."""
    
    def __init__(self, threshold: float = 0.3, min_samples: int = 5):
        """
        Initialize network constructor.
        
        Args:
            threshold: Correlation threshold for network edges
            min_samples: Minimum samples required for network construction
        """
        self.threshold = threshold
        self.min_samples = min_samples
        self.networks = {}
        
    def build_correlation_network(self, features: pd.DataFrame, 
                                method: str = 'spearman') -> nx.Graph:
        """
        Build correlation-based network from feature matrix.
        
        Args:
            features: Feature matrix (samples x features)
            method: Correlation method ('spearman', 'pearson')
            
        Returns:
            NetworkX graph with correlation edges
        """
        logger.info(f"Building {method} correlation network...")
        
        # Calculate correlation matrix
        if method == 'spearman':
            corr_matrix = features.corr(method='spearman')
        else:
            corr_matrix = features.corr(method='pearson')
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (features)
        for feature in features.columns:
            G.add_node(feature, feature_type=self._get_feature_type(feature))
        
        # Add edges based on correlation threshold
        for i, feature1 in enumerate(features.columns):
            for j, feature2 in enumerate(features.columns[i+1:], i+1):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= self.threshold:
                    G.add_edge(feature1, feature2, 
                             weight=abs(corr_val),
                             correlation=corr_val,
                             edge_type='correlation')
        
        logger.info(f"Created {method} network: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        
        self.networks[f'{method}_correlation'] = G
        return G
    
    def build_cooccurrence_network(self, features: pd.DataFrame) -> nx.Graph:
        """
        Build co-occurrence network based on presence/absence patterns.
        
        Args:
            features: Feature matrix (samples x features)
            
        Returns:
            NetworkX graph with co-occurrence edges
        """
        logger.info("Building co-occurrence network...")
        
        # Binarize features (presence/absence)
        binary_features = (features > 0).astype(int)
        
        # Calculate Jaccard similarity for co-occurrence
        G = nx.Graph()
        
        # Add nodes
        for feature in features.columns:
            prevalence = (features[feature] > 0).sum() / len(features)
            G.add_node(feature, 
                      feature_type=self._get_feature_type(feature),
                      prevalence=prevalence)
        
        # Calculate co-occurrence for each pair
        for i, feature1 in enumerate(features.columns):
            for j, feature2 in enumerate(features.columns[i+1:], i+1):
                # Jaccard similarity
                intersection = (binary_features[feature1] & 
                              binary_features[feature2]).sum()
                union = (binary_features[feature1] | 
                        binary_features[feature2]).sum()
                
                if union > 0:
                    jaccard = intersection / union
                    if jaccard >= self.threshold:
                        G.add_edge(feature1, feature2,
                                 weight=jaccard,
                                 jaccard=jaccard,
                                 edge_type='cooccurrence')
        
        logger.info(f"Created co-occurrence network: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        
        self.networks['cooccurrence'] = G
        return G
    
    def build_functional_network(self, features: pd.DataFrame) -> nx.Graph:
        """
        Build functional pathway network based on KEGG annotations.
        
        Args:
            features: Feature matrix with functional annotations
            
        Returns:
            NetworkX graph with functional relationships
        """
        logger.info("Building functional pathway network...")
        
        G = nx.Graph()
        
        # Group features by functional categories
        function_groups = self._group_by_function(features.columns)
        
        # Add nodes with functional annotations
        for feature in features.columns:
            func_category = self._get_functional_category(feature)
            G.add_node(feature,
                      feature_type=self._get_feature_type(feature),
                      functional_category=func_category)
        
        # Add edges between functionally related features
        for category, features_in_category in function_groups.items():
            if len(features_in_category) > 1:
                # Connect all features within the same functional category
                for i, feat1 in enumerate(features_in_category):
                    for feat2 in features_in_category[i+1:]:
                        # Calculate functional similarity
                        similarity = self._calculate_functional_similarity(
                            features[feat1], features[feat2])
                        
                        if similarity >= self.threshold:
                            G.add_edge(feat1, feat2,
                                     weight=similarity,
                                     functional_similarity=similarity,
                                     edge_type='functional',
                                     pathway=category)
        
        logger.info(f"Created functional network: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        
        self.networks['functional'] = G
        return G
    
    def build_temporal_network(self, features_t1: pd.DataFrame, 
                             features_t2: pd.DataFrame) -> nx.Graph:
        """
        Build temporal transition network from T1 to T2 changes.
        
        Args:
            features_t1: Features at timepoint T1
            features_t2: Features at timepoint T2
            
        Returns:
            NetworkX graph with temporal transitions
        """
        logger.info("Building temporal transition network...")
        
        # Calculate temporal changes
        temporal_changes = features_t2 - features_t1
        
        G = nx.Graph()
        
        # Add nodes with temporal dynamics
        for feature in features_t1.columns:
            stability = 1 - np.std(temporal_changes[feature]) / \
                       (np.mean(np.abs(features_t1[feature])) + 1e-8)
            
            G.add_node(feature,
                      feature_type=self._get_feature_type(feature),
                      temporal_stability=stability,
                      mean_change=np.mean(temporal_changes[feature]))
        
        # Add edges based on correlated temporal changes
        change_corr = temporal_changes.corr(method='spearman')
        
        for i, feature1 in enumerate(features_t1.columns):
            for j, feature2 in enumerate(features_t1.columns[i+1:], i+1):
                corr_val = change_corr.iloc[i, j]
                if abs(corr_val) >= self.threshold:
                    G.add_edge(feature1, feature2,
                             weight=abs(corr_val),
                             temporal_correlation=corr_val,
                             edge_type='temporal')
        
        logger.info(f"Created temporal network: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        
        self.networks['temporal'] = G
        return G
    
    def build_multilayer_network(self, networks: List[nx.Graph]) -> nx.Graph:
        """
        Combine multiple networks into a multilayer representation.
        
        Args:
            networks: List of NetworkX graphs to combine
            
        Returns:
            Combined multilayer network
        """
        logger.info("Building multilayer network...")
        
        G_combined = nx.Graph()
        
        # Collect all nodes
        all_nodes = set()
        for net in networks:
            all_nodes.update(net.nodes())
        
        # Add nodes with aggregated attributes
        for node in all_nodes:
            node_attrs = {}
            for net in networks:
                if node in net.nodes():
                    node_attrs.update(net.nodes[node])
            G_combined.add_node(node, **node_attrs)
        
        # Add edges with layer information
        for i, net in enumerate(networks):
            layer_name = list(self.networks.keys())[i] if i < len(self.networks) else f'layer_{i}'
            
            for u, v, attrs in net.edges(data=True):
                if G_combined.has_edge(u, v):
                    # Combine edge weights
                    existing_weight = G_combined[u][v].get('weight', 0)
                    new_weight = existing_weight + attrs.get('weight', 0)
                    G_combined[u][v]['weight'] = new_weight
                    G_combined[u][v][f'{layer_name}_weight'] = attrs.get('weight', 0)
                else:
                    attrs[f'{layer_name}_weight'] = attrs.get('weight', 0)
                    G_combined.add_edge(u, v, **attrs)
        
        logger.info(f"Created multilayer network: {G_combined.number_of_nodes()} nodes, "
                   f"{G_combined.number_of_edges()} edges")
        
        self.networks['multilayer'] = G_combined
        return G_combined
    
    def _get_feature_type(self, feature_name: str) -> str:
        """Determine feature type from name."""
        if 'species_' in feature_name:
            return 'species'
        elif 'function_' in feature_name:
            return 'function'
        elif 'pca_' in feature_name:
            return 'pca'
        elif any(x in feature_name for x in ['temporal_', 'change_', 'stability_']):
            return 'temporal'
        else:
            return 'other'
    
    def _group_by_function(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Group features by functional categories."""
        groups = {}
        for feature in feature_names:
            if 'function_K' in feature:
                # Extract KEGG ortholog number
                ko_num = feature.split('function_K')[1].split('_')[0] if 'function_K' in feature else 'unknown'
                pathway = self._kegg_to_pathway(ko_num)
                if pathway not in groups:
                    groups[pathway] = []
                groups[pathway].append(feature)
            else:
                category = self._get_feature_type(feature)
                if category not in groups:
                    groups[category] = []
                groups[category].append(feature)
        return groups
    
    def _get_functional_category(self, feature_name: str) -> str:
        """Get functional category for a feature."""
        if 'function_K' in feature_name:
            ko_num = feature_name.split('function_K')[1].split('_')[0]
            return self._kegg_to_pathway(ko_num)
        return self._get_feature_type(feature_name)
    
    def _kegg_to_pathway(self, ko_number: str) -> str:
        """Map KEGG ortholog to pathway (simplified mapping)."""
        # Simplified pathway mapping - in real implementation,
        # this would use KEGG database
        pathway_map = {
            '00': 'Metabolism',
            '01': 'Genetic Information Processing',
            '02': 'Environmental Information Processing',
            '03': 'Cellular Processes',
            '04': 'Organismal Systems',
            '05': 'Human Diseases'
        }
        
        if len(ko_number) >= 2:
            category = ko_number[:2]
            return pathway_map.get(category, 'Unknown')
        return 'Unknown'
    
    def _calculate_functional_similarity(self, feat1: pd.Series, 
                                       feat2: pd.Series) -> float:
        """Calculate functional similarity between two features."""
        # Use Spearman correlation as functional similarity
        correlation, _ = stats.spearmanr(feat1, feat2)
        return abs(correlation) if not np.isnan(correlation) else 0.0

class GraphNeuralNetworkModel(nn.Module):
    """
    Comprehensive GNN model with multiple architectures.
    """
    
    def __init__(self, num_features: int, num_classes: int, 
                 architecture: str = 'gcn', hidden_dim: int = 64,
                 num_layers: int = 3, dropout: float = 0.2,
                 pooling: str = 'mean', use_attention: bool = False):
        """
        Initialize GNN model.
        
        Args:
            num_features: Number of input node features
            num_classes: Number of output classes
            architecture: GNN architecture ('gcn', 'gat', 'sage', 'transformer')
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            pooling: Graph pooling method ('mean', 'max', 'add', 'topk')
            use_attention: Whether to use attention mechanisms
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.use_attention = use_attention
        
        # Input projection
        self.input_projection = nn.Linear(num_features, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if architecture == 'gcn':
                layer = GCNConv(hidden_dim, hidden_dim)
            elif architecture == 'gat':
                layer = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            elif architecture == 'sage':
                layer = SAGEConv(hidden_dim, hidden_dim)
            elif architecture == 'transformer':
                layer = TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            self.gnn_layers.append(layer)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Pooling layer
        if pooling == 'topk':
            self.pool = TopKPooling(hidden_dim, ratio=0.5)
        elif pooling == 'sag':
            self.pool = SAGPooling(hidden_dim, ratio=0.5)
        # For 'mean', 'max', 'add', we'll use functional pooling
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        """Forward pass through the GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x_new = layer(x, edge_index)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # Graph-level pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling in ['topk', 'sag']:
            x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
            x = global_mean_pool(x, batch)
        
        # Attention mechanism
        if self.use_attention:
            # Reshape for attention
            x = x.unsqueeze(1)  # Add sequence dimension
            x, _ = self.attention(x, x, x)
            x = x.squeeze(1)  # Remove sequence dimension
        
        # Classification
        out = self.classifier(x)
        
        return out

class TemporalGraphTransformer(nn.Module):
    """
    Temporal Graph Transformer for analyzing T1/T2 dynamics.
    """
    
    def __init__(self, num_features: int, num_classes: int,
                 hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embeddings
        self.node_embedding = nn.Linear(num_features, hidden_dim)
        
        # Temporal encoding
        self.temporal_embedding = nn.Embedding(2, hidden_dim)  # T1, T2
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
            for _ in range(num_layers)
        ])
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data_t1, data_t2):
        """Forward pass with temporal data."""
        # Process T1 data
        x_t1 = self.node_embedding(data_t1.x)
        temp_emb_t1 = self.temporal_embedding(torch.zeros(x_t1.size(0), dtype=torch.long, device=x_t1.device))
        x_t1 = x_t1 + temp_emb_t1
        
        for layer in self.transformer_layers:
            x_t1 = layer(x_t1, data_t1.edge_index)
            x_t1 = F.relu(x_t1)
        
        # Process T2 data
        x_t2 = self.node_embedding(data_t2.x)
        temp_emb_t2 = self.temporal_embedding(torch.ones(x_t2.size(0), dtype=torch.long, device=x_t2.device))
        x_t2 = x_t2 + temp_emb_t2
        
        for layer in self.transformer_layers:
            x_t2 = layer(x_t2, data_t2.edge_index)
            x_t2 = F.relu(x_t2)
        
        # Temporal attention between T1 and T2
        x_temporal = torch.stack([
            global_mean_pool(x_t1, data_t1.batch),
            global_mean_pool(x_t2, data_t2.batch)
        ], dim=1)
        
        x_attended, _ = self.temporal_attention(x_temporal, x_temporal, x_temporal)
        x_final = x_attended.mean(dim=1)
        
        # Classification
        out = self.classifier(x_final)
        return out

class GraphContrastiveLearning(nn.Module):
    """
    Graph contrastive learning for self-supervised representation learning.
    """
    
    def __init__(self, encoder: nn.Module, hidden_dim: int = 64, 
                 temperature: float = 0.07):
        super().__init__()
        
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.temperature = temperature
    
    def forward(self, data1, data2):
        """Contrastive forward pass."""
        # Encode both augmented views
        z1 = self.encoder(data1)
        z2 = self.encoder(data2)
        
        # Project to contrastive space
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        
        return p1, p2
    
    def contrastive_loss(self, z1, z2):
        """Compute contrastive loss."""
        # Normalize representations
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        
        # Labels for positive pairs
        labels = torch.arange(z1.size(0), device=z1.device)
        
        # Contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class MicrobiomeGNNPipeline:
    """
    Comprehensive pipeline for GNN-based microbiome analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GNN pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network_constructor = MicrobiomeNetworkConstructor(
            threshold=config.get('network_threshold', 0.3)
        )
        self.results = {}
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'gnn_outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized GNN pipeline with device: {self.device}")
    
    def load_data(self, features_path: str, metadata_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load microbiome features and metadata."""
        logger.info(f"Loading data from {features_path} and {metadata_path}")
        
        features = pd.read_csv(features_path, index_col=0)
        metadata = pd.read_csv(metadata_path, index_col=0)
        
        # Align data
        common_samples = features.index.intersection(metadata.index)
        features = features.loc[common_samples]
        metadata = metadata.loc[common_samples]
        
        logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
        return features, metadata
    
    def construct_networks(self, features: pd.DataFrame) -> Dict[str, nx.Graph]:
        """Construct multiple types of microbiome networks."""
        logger.info("Constructing microbiome networks...")
        
        networks = {}
        
        # Correlation networks
        networks['spearman'] = self.network_constructor.build_correlation_network(
            features, method='spearman'
        )
        networks['pearson'] = self.network_constructor.build_correlation_network(
            features, method='pearson'
        )
        
        # Co-occurrence network
        networks['cooccurrence'] = self.network_constructor.build_cooccurrence_network(features)
        
        # Functional network
        networks['functional'] = self.network_constructor.build_functional_network(features)
        
        # Multilayer network
        networks['multilayer'] = self.network_constructor.build_multilayer_network(
            list(networks.values())
        )
        
        return networks
    
    def prepare_graph_data(self, features: pd.DataFrame, metadata: pd.DataFrame,
                          network: nx.Graph) -> List[Data]:
        """Prepare graph data for PyTorch Geometric."""
        logger.info("Preparing graph data for PyTorch Geometric...")
        
        graph_data_list = []
        
        # Node features - use the original feature values
        node_features = features.values.T  # Transpose to get features x samples
        
        # Create edge index from network
        edge_list = list(network.edges())
        if not edge_list:
            logger.warning("No edges found in network, creating minimal structure")
            # Create a minimal structure if no edges
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        else:
            # Map feature names to indices
            feature_to_idx = {feat: idx for idx, feat in enumerate(features.columns)}
            
            # Filter edges that exist in our feature set
            valid_edges = []
            for u, v in edge_list:
                if u in feature_to_idx and v in feature_to_idx:
                    valid_edges.append([feature_to_idx[u], feature_to_idx[v]])
                    valid_edges.append([feature_to_idx[v], feature_to_idx[u]])  # Undirected
            
            if not valid_edges:
                logger.warning("No valid edges found, creating minimal structure")
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
            else:
                edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
        
        # Create graph data for each sample
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(metadata['symptom'])
        
        for i, (sample_id, sample_features) in enumerate(features.iterrows()):
            # Node features for this sample
            x = torch.tensor(sample_features.values, dtype=torch.float32).unsqueeze(1)
            
            # Create graph data
            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(labels[i], dtype=torch.long),
                sample_id=sample_id
            )
            
            graph_data_list.append(data)
        
        logger.info(f"Created {len(graph_data_list)} graph samples")
        return graph_data_list, label_encoder
    
    def train_gnn_model(self, graph_data: List[Data], architecture: str = 'gcn',
                       num_epochs: int = 100, batch_size: int = 32) -> nn.Module:
        """Train GNN model with cross-validation."""
        logger.info(f"Training {architecture} model...")
        
        # Split data
        train_size = int(0.8 * len(graph_data))
        train_data = graph_data[:train_size]
        val_data = graph_data[train_size:]
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        num_features = graph_data[0].x.size(1)
        num_classes = len(set([data.y.item() for data in graph_data]))
        
        model = GraphNeuralNetworkModel(
            num_features=num_features,
            num_classes=num_classes,
            architecture=architecture,
            hidden_dim=self.config.get('hidden_dim', 64),
            num_layers=self.config.get('num_layers', 3),
            dropout=self.config.get('dropout', 0.2),
            pooling=self.config.get('pooling', 'mean')
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=self.config.get('learning_rate', 0.001))
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch)
                loss = criterion(out, batch.y)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            val_acc = correct / total
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.output_dir / f'best_{architecture}_model.pth')
            
            scheduler.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc
        }
        
        with open(self.output_dir / f'{architecture}_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        return model, history
    
    def evaluate_model(self, model: nn.Module, graph_data: List[Data], 
                      label_encoder: LabelEncoder) -> Dict[str, Any]:
        """Evaluate GNN model performance."""
        logger.info("Evaluating model performance...")
        
        # Create data loader
        data_loader = DataLoader(graph_data, batch_size=32, shuffle=False)
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = model(batch)
                pred = out.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # Convert back to original labels
        pred_labels = label_encoder.inverse_transform(all_predictions)
        true_labels = label_encoder.inverse_transform(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        classification_rep = classification_report(true_labels, pred_labels, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': pred_labels.tolist(),
            'true_labels': true_labels.tolist()
        }
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        return results
    
    def analyze_graph_patterns(self, networks: Dict[str, nx.Graph]) -> Dict[str, Any]:
        """Analyze graph topology and biological patterns."""
        logger.info("Analyzing graph patterns and topology...")
        
        analysis_results = {}
        
        for network_name, network in networks.items():
            logger.info(f"Analyzing {network_name} network...")
            
            # Basic topology metrics
            topology = {
                'num_nodes': network.number_of_nodes(),
                'num_edges': network.number_of_edges(),
                'density': nx.density(network),
                'average_clustering': nx.average_clustering(network),
                'num_components': nx.number_connected_components(network)
            }
            
            # Centrality measures
            centrality = {
                'degree_centrality': nx.degree_centrality(network),
                'betweenness_centrality': nx.betweenness_centrality(network),
                'eigenvector_centrality': nx.eigenvector_centrality(network, max_iter=1000),
                'pagerank': nx.pagerank(network)
            }
            
            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(network)
                modularity = nx.community.modularity(network, communities)
                
                community_info = {
                    'num_communities': len(communities),
                    'modularity': modularity,
                    'community_sizes': [len(c) for c in communities]
                }
            except:
                community_info = {'error': 'Community detection failed'}
            
            # Hub identification (top 10 by degree)
            degrees = dict(network.degree())
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            
            analysis_results[network_name] = {
                'topology': topology,
                'centrality': centrality,
                'communities': community_info,
                'top_hubs': top_hubs
            }
        
        return analysis_results
    
    def visualize_networks(self, networks: Dict[str, nx.Graph]):
        """Create network visualizations."""
        logger.info("Creating network visualizations...")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (network_name, network) in enumerate(networks.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Limit network size for visualization
            if network.number_of_nodes() > 100:
                # Take largest connected component
                largest_cc = max(nx.connected_components(network), key=len)
                network_vis = network.subgraph(largest_cc).copy()
                # Further sample if still too large
                if len(network_vis) > 100:
                    nodes_to_keep = list(network_vis.nodes())[:100]
                    network_vis = network_vis.subgraph(nodes_to_keep)
            else:
                network_vis = network
            
            # Create layout
            try:
                pos = nx.spring_layout(network_vis, k=1/np.sqrt(len(network_vis)), iterations=50)
            except:
                pos = nx.random_layout(network_vis)
            
            # Node colors based on feature type
            node_colors = []
            for node in network_vis.nodes():
                if 'species_' in str(node):
                    node_colors.append('lightcoral')
                elif 'function_' in str(node):
                    node_colors.append('lightblue')
                elif 'pca_' in str(node):
                    node_colors.append('lightgreen')
                else:
                    node_colors.append('lightgray')
            
            # Draw network
            nx.draw_networkx_nodes(network_vis, pos, node_color=node_colors, 
                                 node_size=30, alpha=0.7, ax=ax)
            nx.draw_networkx_edges(network_vis, pos, alpha=0.3, width=0.5, ax=ax)
            
            ax.set_title(f'{network_name.title()} Network\n'
                        f'Nodes: {network.number_of_nodes()}, '
                        f'Edges: {network.number_of_edges()}')
            ax.axis('off')
        
        # Remove empty subplots
        for i in range(len(networks), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_visualizations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Network visualizations saved")
    
    def run_comprehensive_analysis(self, features_path: str, metadata_path: str):
        """Run comprehensive GNN analysis pipeline."""
        logger.info("Starting comprehensive GNN analysis...")
        
        # Load data
        features, metadata = self.load_data(features_path, metadata_path)
        
        # Construct networks
        networks = self.construct_networks(features)
        
        # Analyze graph patterns
        graph_analysis = self.analyze_graph_patterns(networks)
        
        # Visualize networks
        self.visualize_networks(networks)
        
        # Test different GNN architectures
        architectures = ['gcn', 'gat', 'sage', 'transformer']
        architecture_results = {}
        
        for arch in architectures:
            logger.info(f"Testing {arch} architecture...")
            
            # Use multilayer network for comprehensive analysis
            graph_data, label_encoder = self.prepare_graph_data(
                features, metadata, networks['multilayer']
            )
            
            # Train model
            model, history = self.train_gnn_model(
                graph_data, architecture=arch,
                num_epochs=self.config.get('num_epochs', 100)
            )
            
            # Evaluate model
            results = self.evaluate_model(model, graph_data, label_encoder)
            
            architecture_results[arch] = {
                'training_history': history,
                'evaluation_results': results
            }
        
        # Compile comprehensive results
        comprehensive_results = {
            'data_info': {
                'num_samples': len(features),
                'num_features': len(features.columns),
                'classes': metadata['symptom'].value_counts().to_dict()
            },
            'network_analysis': graph_analysis,
            'architecture_comparison': architecture_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open(self.output_dir / 'comprehensive_gnn_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Create summary report
        self.create_summary_report(comprehensive_results)
        
        logger.info("Comprehensive GNN analysis completed!")
        return comprehensive_results
    
    def create_summary_report(self, results: Dict[str, Any]):
        """Create a summary report of GNN analysis."""
        report_path = self.output_dir / 'gnn_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Graph Neural Networks Analysis Report\n\n")
            f.write(f"Generated on: {results['timestamp']}\n\n")
            
            # Data overview
            f.write("## Data Overview\n\n")
            data_info = results['data_info']
            f.write(f"- **Samples**: {data_info['num_samples']}\n")
            f.write(f"- **Features**: {data_info['num_features']}\n")
            f.write(f"- **Classes**: {data_info['classes']}\n\n")
            
            # Network analysis
            f.write("## Network Analysis\n\n")
            for network_name, analysis in results['network_analysis'].items():
                f.write(f"### {network_name.title()} Network\n\n")
                topology = analysis['topology']
                f.write(f"- **Nodes**: {topology['num_nodes']}\n")
                f.write(f"- **Edges**: {topology['num_edges']}\n")
                f.write(f"- **Density**: {topology['density']:.4f}\n")
                f.write(f"- **Clustering**: {topology['average_clustering']:.4f}\n")
                f.write(f"- **Components**: {topology['num_components']}\n\n")
            
            # Architecture comparison
            f.write("## GNN Architecture Comparison\n\n")
            f.write("| Architecture | Best Accuracy | Final Accuracy |\n")
            f.write("|--------------|---------------|----------------|\n")
            
            for arch, arch_results in results['architecture_comparison'].items():
                best_acc = arch_results['training_history']['best_val_accuracy']
                final_acc = arch_results['evaluation_results']['accuracy']
                f.write(f"| {arch.upper()} | {best_acc:.4f} | {final_acc:.4f} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Find best architecture
            best_arch = max(results['architecture_comparison'].items(),
                          key=lambda x: x[1]['evaluation_results']['accuracy'])
            
            f.write(f"- **Best performing architecture**: {best_arch[0].upper()} "
                   f"with {best_arch[1]['evaluation_results']['accuracy']:.4f} accuracy\n")
            
            # Network insights
            multilayer_analysis = results['network_analysis'].get('multilayer', {})
            if 'topology' in multilayer_analysis:
                topo = multilayer_analysis['topology']
                f.write(f"- **Network complexity**: {topo['num_nodes']} nodes, "
                       f"{topo['num_edges']} edges\n")
                f.write(f"- **Network density**: {topo['density']:.4f}\n")
                f.write(f"- **Clustering coefficient**: {topo['average_clustering']:.4f}\n")
            
            f.write("\n## Innovation Potential\n\n")
            f.write("This GNN approach provides:\n")
            f.write("- Network-based microbiome interaction modeling\n")
            f.write("- Multi-scale graph representations\n")
            f.write("- Biological pathway integration\n")
            f.write("- Temporal dynamics analysis capability\n")
            f.write("- Interpretable graph patterns for biological insights\n")
        
        logger.info(f"Summary report saved to {report_path}")

def main():
    """Main function to run GNN pipeline."""
    parser = argparse.ArgumentParser(description='GNN Pipeline for Microbiome Analysis')
    parser.add_argument('--features', required=True, help='Path to features CSV')
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV')
    parser.add_argument('--output-dir', default='gnn_outputs', help='Output directory')
    parser.add_argument('--config', help='Path to configuration JSON')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'network_threshold': 0.3,
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'pooling': 'mean'
        }
    
    config['output_dir'] = args.output_dir
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = MicrobiomeGNNPipeline(config)
    results = pipeline.run_comprehensive_analysis(args.features, args.metadata)
    
    print(f"\nGNN Analysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best accuracy achieved: {max([r['evaluation_results']['accuracy'] for r in results['architecture_comparison'].values()]):.4f}")

if __name__ == "__main__":
    main()