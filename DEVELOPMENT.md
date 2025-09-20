# MPEG-G Microbiome Challenge - Track 1: Cytokine Prediction

## Project Overview
This repository contains code for the MPEG-G Microbiome Challenge Track 1, focusing on predicting cytokine levels from microbiome composition data.

## Critical Data Finding
**IMPORTANT**: The microbiome and cytokine datasets are from separate studies and cannot be directly merged:
- **Microbiome Data**: 40 samples, 9,132 features (species + functions)
- **Cytokine Data**: 670 samples, 66 cytokine measurements

## Current Status
- Data exploration and preprocessing completed by sub-agent
- Two independent, analysis-ready datasets created
- Train/test splits prepared for both datasets

## Core Architecture

### Available Scripts Analysis
1. **`scripts/preprocess.py`**: Comprehensive FASTQ preprocessing pipeline with parallel processing, OTU clustering, and diversity metrics
2. **`scripts/ml_pipeline.py`**: Complete ML pipeline with multiple models (RF, XGBoost, LightGBM, PyTorch), ensemble methods, and performance evaluation

### Recommended Data Flow
```
processed_data/ → ML pipeline → model_outputs/ → submission
```

## Quick Commands

### Current Recommended Approach (Using Processed Data)
```bash
# Run ML pipeline on microbiome data (classification)
python scripts/ml_pipeline.py \
  --features processed_data/microbiome_features_processed.csv \
  --output model_outputs_microbiome \
  --seed 42

# Run ML pipeline on cytokine data (regression/clustering)  
python scripts/ml_pipeline.py \
  --features processed_data/cytokine_features_processed.csv \
  --output model_outputs_cytokine \
  --seed 42
```

### Original Pipeline (If Using FASTQ Files)
```bash
# Complete pipeline with high performance
./run_pipeline.sh --cores 16 --chunk-size 20000

# Individual preprocessing
python scripts/preprocess.py \
  --input-dir raw_data/fastq \
  --output-dir processed_data \
  --metadata raw_data/Train.csv \
  --cytokines raw_data/cytokine_profiles.csv \
  --chunk-size 10000 \
  --cores 16
```

### Environment Setup
```bash
# Install requirements
pip install -r requirements.txt

# Check if conda environment exists
conda activate microbiome_env  # if available
```

### Debug and Validation
```bash
# Enable verbose logging
python3 -v scripts/preprocess.py [args]

# Monitor logs
tail -f logs/preprocessing_*.log
tail -f logs/ml_pipeline_*.log

# Check dependencies
python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn, scipy; print('All dependencies OK')"
```

## Available Datasets (Already Processed)

### Processed Data Structure
- **Microbiome Data**: 40 samples, 9,132 features (species + functions)
  - Target: Symptom severity classification (Healthy, Mild, Moderate, Severe)
  - Files: `processed_data/microbiome_*`
- **Cytokine Data**: 670 samples, 66 cytokine measurements  
  - Files: `processed_data/cytokine_*`

### Key Files
- `processed_data/microbiome_features_processed.csv` - Normalized microbiome features
- `processed_data/cytokine_features_processed.csv` - Normalized cytokine data
- `processed_data/microbiome_X_train.csv`, `microbiome_X_test.csv` - Train/test splits
- `processed_data/FINAL_SUMMARY.txt` - Comprehensive data analysis

## Directory Structure

### Input Data Structure
```
raw_data/
├── fastq/                    # FASTQ files (*.fastq, *.fq, compressed OK)
├── Train.csv                 # Primary metadata file
├── Train_Subjects.csv        # Alternative metadata file
├── cytokines.csv            # Cytokine measurements (optional)
└── abundance/               # Pre-computed abundance data
```

### Output Structure
```
processed_data/
├── feature_matrices/        # Main ML input files
├── diversity_metrics/       # Alpha/beta diversity calculations
├── otu_tables/             # Taxonomic abundance tables
├── quality_reports/        # Quality control reports
└── temporal_data/          # Time-series features

model_outputs/
├── submission_report.json  # Performance metrics
├── models/                 # Trained model files
├── predictions/           # Prediction results
├── visualizations/        # Performance plots
└── feature_importance.csv # Feature analysis
```

## Scripts Analysis Summary

### `scripts/preprocess.py` - Comprehensive FASTQ Preprocessing
**Key Features**:
- Parallel FASTQ processing with configurable chunk sizes
- Quality filtering and OTU clustering (simplified MD5-based)
- Diversity metrics: Shannon, Simpson, Chao1, Pielou's evenness
- Multi-format output: CSV, HDF5, pickle
- Memory-efficient chunk-based processing
- Log transformation and scaling for cytokine data

**Classes**: `MicrobiomePreprocessor` - Main preprocessing orchestrator

### `scripts/ml_pipeline.py` - Advanced ML Pipeline  
**Key Features**:
- Multiple models: Random Forest, XGBoost, LightGBM, PyTorch Neural Networks
- Multi-modal Transformer architecture for body-site aware modeling
- Advanced feature engineering (log transforms, ratios, interactions)
- Ensemble methods with weighted averaging
- Comprehensive metrics: RMSE, R², MAE
- Feature importance analysis and visualizations
- Cross-validation and hyperparameter optimization

**Classes**: 
- `CytokinePredictionPipeline` - Main ML orchestrator
- `MultiModalTransformer` - PyTorch transformer for multi-site data
- `MicrobiomeCytokineDataset` - PyTorch dataset wrapper

## Current Challenge Strategy

**Track 1**: Cytokine Prediction (Challenge requirement)
- **Problem**: Separate datasets prevent direct microbiome → cytokine prediction
- **Alternative Approach**: Develop methodology on available data that can be applied to future merged datasets
- **Value**: Demonstrates advanced ML techniques applicable to microbiome-cytokine interactions

## System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ recommended (32GB+ for large datasets)
- **Storage**: 20GB+ free space
- **CPU**: Multi-core recommended (4+ cores)

## Error Handling and Troubleshooting

The pipeline includes comprehensive validation:
- Input file checks
- Memory and disk space verification
- Dependency validation
- Output verification
- Performance metrics validation

Common issues and solutions are documented in the main README.md file.

## Cross-Platform Support

The pipeline works on macOS and Linux with automatic platform detection for system resource checking and path handling.