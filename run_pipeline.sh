#!/bin/bash

# MPEG-G Microbiome Challenge - Complete Pipeline Runner
# This script orchestrates the entire preprocessing and ML pipeline

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
INPUT_DIR="./raw_data/fastq"
OUTPUT_DIR="./processed_data"
MODEL_DIR="./model_outputs"
METADATA="./raw_data/Train.csv"
CYTOKINES="./raw_data/cytokines.csv"
CORES=8
CHUNK_SIZE=10000
LOG_DIR="./logs"

# Create log directory
mkdir -p $LOG_DIR

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if required files exist
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Python environment (virtual environment recommended)
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_status "Using virtual environment: $VIRTUAL_ENV"
    else
        print_warning "No virtual environment detected. Consider using one."
    fi
    
    # Check required directories
    if [ ! -d "$INPUT_DIR" ]; then
        print_warning "Input directory $INPUT_DIR does not exist. Creating it..."
        mkdir -p "$INPUT_DIR"
    fi
    
    # Check metadata files
    if [ ! -f "$METADATA" ]; then
        print_warning "Metadata file $METADATA does not exist"
        if [ -f "./raw_data/Train_Subjects.csv" ]; then
            print_info "Found Train_Subjects.csv, will use it as metadata"
            METADATA="./raw_data/Train_Subjects.csv"
        elif [ -f "./raw_data/Train.csv" ]; then
            print_info "Found Train.csv, will use it as metadata"
            METADATA="./raw_data/Train.csv"
        else
            print_error "No metadata file found"
            exit 1
        fi
    fi
    
    # Check for FASTQ files (if input directory exists and has files)
    if [ -d "$INPUT_DIR" ]; then
        FASTQ_COUNT=$(find $INPUT_DIR -name "*.fastq*" -o -name "*.fq*" 2>/dev/null | wc -l)
        if [ $FASTQ_COUNT -eq 0 ]; then
            print_warning "No FASTQ files found in $INPUT_DIR"
            print_info "Pipeline will work with existing processed data if available"
        else
            print_status "Found $FASTQ_COUNT FASTQ files"
        fi
    fi
    
    # Check available disk space (macOS compatible)
    if command -v df &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            AVAILABLE_SPACE=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "20")
        else
            # Linux
            AVAILABLE_SPACE=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "20")
        fi
        
        if [ "$AVAILABLE_SPACE" -lt 20 ] 2>/dev/null; then
            print_warning "Low disk space: ${AVAILABLE_SPACE}GB available. Recommend at least 20GB."
        else
            print_status "Disk space: ${AVAILABLE_SPACE}GB available"
        fi
    fi
    
    # Check available memory (cross-platform)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        TOTAL_MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "8589934592")
        TOTAL_MEM=$((TOTAL_MEM_BYTES / 1024 / 1024 / 1024))
    elif command -v free &> /dev/null; then
        # Linux
        TOTAL_MEM=$(free -g 2>/dev/null | awk 'NR==2 {print $2}' || echo "8")
    else
        TOTAL_MEM=8
    fi
    
    if [ "$TOTAL_MEM" -lt 8 ] 2>/dev/null; then
        print_warning "Low memory: ${TOTAL_MEM}GB available. Recommend at least 8GB."
    else
        print_status "Memory: ${TOTAL_MEM}GB available"
    fi
    
    # Check Python packages
    print_status "Checking Python dependencies..."
    python3 -c "
import sys
required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'scipy']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print('Missing packages:', ', '.join(missing))
    sys.exit(1)
else:
    print('All required packages found')
" || {
        print_error "Missing required Python packages. Install with: pip install -r requirements.txt"
        exit 1
    }
    
    print_status "All requirements satisfied"
}

# Function to run preprocessing
run_preprocessing() {
    print_status "Starting preprocessing pipeline..."
    
    PREPROCESS_LOG="$LOG_DIR/preprocessing_$(date +'%Y%m%d_%H%M%S').log"
    
    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/feature_matrices"
    
    # Build command
    PREPROCESS_CMD="python3 scripts/preprocess.py \
        --input-dir $INPUT_DIR \
        --output-dir $OUTPUT_DIR \
        --metadata $METADATA \
        --chunk-size $CHUNK_SIZE \
        --cores $CORES"
    
    # Add cytokines if file exists
    if [ -f "$CYTOKINES" ]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --cytokines $CYTOKINES"
    fi
    
    print_info "Running: $PREPROCESS_CMD"
    
    # Run preprocessing with logging
    $PREPROCESS_CMD 2>&1 | tee $PREPROCESS_LOG
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_status "Preprocessing completed successfully"
        print_info "Log saved to: $PREPROCESS_LOG"
    else
        print_error "Preprocessing failed. Check $PREPROCESS_LOG for details"
        exit 1
    fi
}

# Function to run ML pipeline
run_ml_pipeline() {
    print_status "Starting ML pipeline..."
    
    # Create model output directory
    mkdir -p "$MODEL_DIR"
    
    # Find the latest feature matrix
    FEATURE_MATRIX=$(ls -t $OUTPUT_DIR/feature_matrices/features_*.csv 2>/dev/null | head -1)
    
    if [ -z "$FEATURE_MATRIX" ]; then
        print_error "No feature matrix found in $OUTPUT_DIR/feature_matrices/"
        print_info "Available files:"
        ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "Output directory is empty"
        exit 1
    fi
    
    print_status "Using feature matrix: $FEATURE_MATRIX"
    
    ML_LOG="$LOG_DIR/ml_pipeline_$(date +'%Y%m%d_%H%M%S').log"
    
    # Build ML command
    ML_CMD="python3 scripts/ml_pipeline.py \
        --features $FEATURE_MATRIX \
        --output $MODEL_DIR \
        --seed 42"
    
    # Add cytokines if processed file exists
    if [ -f "$OUTPUT_DIR/cytokine_data.csv" ]; then
        ML_CMD="$ML_CMD --cytokines $OUTPUT_DIR/cytokine_data.csv"
    elif [ -f "$CYTOKINES" ]; then
        ML_CMD="$ML_CMD --cytokines $CYTOKINES"
    fi
    
    print_info "Running: $ML_CMD"
    
    # Run ML pipeline with logging
    $ML_CMD 2>&1 | tee $ML_LOG
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_status "ML pipeline completed successfully"
        print_info "Log saved to: $ML_LOG"
    else
        print_error "ML pipeline failed. Check $ML_LOG for details"
        exit 1
    fi
}

# Function to generate submission package
create_submission() {
    print_status "Creating submission package..."
    
    TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
    SUBMISSION_DIR="./submission_$TIMESTAMP"
    
    mkdir -p $SUBMISSION_DIR
    
    # Copy essential files if they exist
    if [ -f "$MODEL_DIR/submission_report.json" ]; then
        cp "$MODEL_DIR/submission_report.json" "$SUBMISSION_DIR/"
    fi
    
    if [ -d "$MODEL_DIR/visualizations" ]; then
        cp -r "$MODEL_DIR/visualizations" "$SUBMISSION_DIR/"
    fi
    
    if [ -f "$MODEL_DIR/feature_importance.csv" ]; then
        cp "$MODEL_DIR/feature_importance.csv" "$SUBMISSION_DIR/"
    fi
    
    # Copy code
    mkdir -p "$SUBMISSION_DIR/code"
    cp scripts/*.py "$SUBMISSION_DIR/code/" 2>/dev/null || print_warning "No Python scripts found to copy"
    cp requirements.txt "$SUBMISSION_DIR/" 2>/dev/null || print_warning "No requirements.txt found"
    cp "$0" "$SUBMISSION_DIR/run_pipeline.sh" 2>/dev/null || print_warning "Cannot copy pipeline script"
    
    # Copy logs
    mkdir -p "$SUBMISSION_DIR/logs"
    cp $LOG_DIR/*.log "$SUBMISSION_DIR/logs/" 2>/dev/null || print_warning "No logs found to copy"
    
    # Create documentation
    cat > "$SUBMISSION_DIR/README.md" << EOF
# MPEG-G Challenge Submission

## Track 1: Cytokine Prediction from Microbiome Data

### Files Included
- submission_report.json: Model performance metrics
- visualizations/: Performance visualizations
- feature_importance.csv: Feature importance analysis
- code/: Complete preprocessing and ML pipeline code
- logs/: Processing logs

### Best Model Performance
Check submission_report.json for detailed metrics

### How to Reproduce
1. Install requirements: pip install -r requirements.txt
2. Run complete pipeline: ./run_pipeline.sh
3. Or run components separately:
   - Preprocessing: python code/preprocess.py [args]
   - ML pipeline: python code/ml_pipeline.py [args]

### Pipeline Commands Used
- Preprocessing: python scripts/preprocess.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --metadata $METADATA --cores $CORES
- ML Pipeline: python scripts/ml_pipeline.py --features [feature_matrix] --output $MODEL_DIR --seed 42

### System Information
- Generated on: $TIMESTAMP
- Input directory: $INPUT_DIR
- Output directory: $OUTPUT_DIR
- Cores used: $CORES
- Chunk size: $CHUNK_SIZE

### Processing Summary
Check logs/ directory for detailed processing information.

EOF
    
    # Create ZIP file if zip is available
    if command -v zip &> /dev/null; then
        zip -r "submission_$TIMESTAMP.zip" "$SUBMISSION_DIR"
        print_status "Submission package created: submission_$TIMESTAMP.zip"
    else
        print_status "Submission directory created: $SUBMISSION_DIR"
        print_warning "zip command not found. Package not compressed."
    fi
}

# Function to run quick validation
validate_results() {
    print_status "Validating results..."
    
    # Check if submission report exists
    if [ -f "$MODEL_DIR/submission_report.json" ]; then
        # Extract RMSE from report
        RMSE=$(python3 -c "
import json
try:
    with open('$MODEL_DIR/submission_report.json') as f:
        data = json.load(f)
    print(data.get('best_rmse', 'N/A'))
except:
    print('N/A')
" 2>/dev/null)
        print_status "Best RMSE achieved: $RMSE"
    else
        print_warning "No submission report found"
    fi
    
    # Check feature matrix size
    if [ -f "$FEATURE_MATRIX" ]; then
        if command -v du &> /dev/null; then
            MATRIX_SIZE=$(du -h "$FEATURE_MATRIX" 2>/dev/null | cut -f1 || echo "Unknown")
            print_status "Feature matrix size: $MATRIX_SIZE"
        fi
    fi
    
    # Check model outputs
    if [ -d "$MODEL_DIR" ]; then
        MODEL_COUNT=$(ls "$MODEL_DIR"/*.pkl 2>/dev/null | wc -l || echo "0")
        print_status "Models saved: $MODEL_COUNT"
    fi
    
    # Check visualization outputs
    if [ -d "$MODEL_DIR/visualizations" ]; then
        VIZ_COUNT=$(ls "$MODEL_DIR/visualizations"/*.png 2>/dev/null | wc -l || echo "0")
        print_status "Visualizations created: $VIZ_COUNT"
    fi
}

# Function to show help
show_help() {
    echo "MPEG-G Microbiome Challenge Pipeline Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR      Input directory with FASTQ files (default: ./raw_data/fastq)"
    echo "  --output-dir DIR     Output directory for processed data (default: ./processed_data)"
    echo "  --model-dir DIR      Output directory for models (default: ./model_outputs)"
    echo "  --metadata FILE      Path to metadata CSV file (default: ./raw_data/Train.csv)"
    echo "  --cytokines FILE     Path to cytokines CSV file (default: ./raw_data/cytokines.csv)"
    echo "  --cores N           Number of CPU cores to use (default: 8)"
    echo "  --chunk-size N      Chunk size for processing (default: 10000)"
    echo "  --skip-preprocess   Skip preprocessing step"
    echo "  --skip-ml          Skip ML pipeline"
    echo "  --skip-validation  Skip result validation"
    echo "  --auto-submit      Automatically create submission package"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with defaults"
    echo "  $0 --cores 16 --chunk-size 20000    # Use more resources"
    echo "  $0 --skip-preprocess                # Only run ML pipeline"
    echo "  $0 --auto-submit                    # Run everything and create submission"
}

# Main execution flow
main() {
    echo "======================================"
    echo "MPEG-G Microbiome Challenge Pipeline"
    echo "======================================"
    echo ""
    
    # Initialize flags
    SKIP_PREPROCESS=false
    SKIP_ML=false
    SKIP_VALIDATION=false
    AUTO_SUBMIT=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input-dir)
                INPUT_DIR="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --model-dir)
                MODEL_DIR="$2"
                shift 2
                ;;
            --metadata)
                METADATA="$2"
                shift 2
                ;;
            --cytokines)
                CYTOKINES="$2"
                shift 2
                ;;
            --cores)
                CORES="$2"
                shift 2
                ;;
            --chunk-size)
                CHUNK_SIZE="$2"
                shift 2
                ;;
            --skip-preprocess)
                SKIP_PREPROCESS=true
                shift
                ;;
            --skip-ml)
                SKIP_ML=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --auto-submit)
                AUTO_SUBMIT=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Show configuration
    print_info "Configuration:"
    print_info "  Input directory: $INPUT_DIR"
    print_info "  Output directory: $OUTPUT_DIR"
    print_info "  Model directory: $MODEL_DIR"
    print_info "  Metadata file: $METADATA"
    print_info "  Cytokines file: $CYTOKINES"
    print_info "  CPU cores: $CORES"
    print_info "  Chunk size: $CHUNK_SIZE"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Run preprocessing
    if [ "$SKIP_PREPROCESS" != true ]; then
        START_TIME=$(date +%s)
        run_preprocessing
        END_TIME=$(date +%s)
        PREPROCESS_TIME=$((END_TIME - START_TIME))
        print_status "Preprocessing took $((PREPROCESS_TIME / 60)) minutes $((PREPROCESS_TIME % 60)) seconds"
    else
        print_status "Skipping preprocessing (--skip-preprocess flag set)"
    fi
    
    # Run ML pipeline
    if [ "$SKIP_ML" != true ]; then
        START_TIME=$(date +%s)
        run_ml_pipeline
        END_TIME=$(date +%s)
        ML_TIME=$((END_TIME - START_TIME))
        print_status "ML pipeline took $((ML_TIME / 60)) minutes $((ML_TIME % 60)) seconds"
    else
        print_status "Skipping ML pipeline (--skip-ml flag set)"
    fi
    
    # Validate results
    if [ "$SKIP_VALIDATION" != true ]; then
        validate_results
    else
        print_status "Skipping validation (--skip-validation flag set)"
    fi
    
    # Create submission package
    if [ "$AUTO_SUBMIT" = true ]; then
        create_submission
    else
        echo ""
        read -p "Create submission package? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            create_submission
        fi
    fi
    
    echo ""
    echo "======================================"
    echo "Pipeline Completed Successfully!"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "1. Review results in $MODEL_DIR"
    if [ -d "$MODEL_DIR/visualizations" ]; then
        echo "2. Check visualizations in $MODEL_DIR/visualizations/"
    fi
    echo "3. Review logs in $LOG_DIR"
    echo "4. Upload processed files for further analysis"
    echo ""
    print_status "All tasks completed!"
}

# Run main function with all arguments
main "$@"