#!/usr/bin/bash -l
#SBATCH --job-name=ifs_bulk_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=pp-long
#SBATCH --account=s83
#SBATCH --output=logs/out_ifs_bulk_%j.log
#SBATCH --error=logs/err_ifs_bulk_%j.log
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue

# Configuration
export OUTPUT_DIR="/capstor/store/cscs/swissai/a122/IFS"
export MODEL_NAME="graphcast"  # or "fourcastnetv2-small"
export INTERVAL=6
export DOWNLOAD_TYPE="both"  # "ensemble", "control", or "both"

# Date ranges to download (8 weeks total)
export DATE_RANGES=(
    "2023-01-02T00:2023-01-08T23"
    "2023-04-02T00:2023-04-08T23" 
    "2023-07-02T00:2023-07-08T23"
    "2023-10-02T00:2023-10-08T23"
    "2024-01-02T00:2024-01-08T23"
    "2024-04-02T00:2024-04-08T23"
    "2024-07-02T00:2024-07-08T23"
    "2024-10-02T00:2024-10-08T23"
)

# Create logs directory
mkdir -p logs

# Load conda/mamba environment
if command -v mamba &> /dev/null; then
    source "$(mamba info --base)/etc/profile.d/mamba.sh"
elif command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Neither conda nor mamba found, trying to source from common locations"
    if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/opt/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
        source "/usr/local/miniconda3/etc/profile.d/conda.sh"
    fi
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q ai_models_ens; then
    echo "Creating conda environment ai_models_ens..."
    if [ -f "environment.yml" ]; then
        mamba env create -n ai_models_ens -f environment.yml || conda env create -n ai_models_ens -f environment.yml
    else
        echo "environment.yml not found, creating basic environment..."
        conda create -n ai_models_ens python=3.9 -y
        conda activate ai_models_ens
        pip install earthkit-data xarray zarr dask
    fi
fi

# Activate environment
conda activate ai_models_ens

echo "========================================================"
echo "Starting bulk IFS download job"
echo "========================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Download type: $DOWNLOAD_TYPE"
echo "Interval: ${INTERVAL}h"
echo "Number of date ranges: ${#DATE_RANGES[@]}"
echo "========================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy source files for reproducibility
mkdir -p "$OUTPUT_DIR/source_files"
cp -r * "$OUTPUT_DIR/source_files/" 2>/dev/null || true

# Run the bulk download script
srun /capstor/store/cscs/swissai/a122/IFS/repo-download-ifs/.venv/bin/python download_ifs_bulk.py \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --interval "$INTERVAL" \
    --download-type "$DOWNLOAD_TYPE" \
    --date-ranges "${DATE_RANGES[@]}"

echo "========================================================"
echo "Bulk download job completed"
echo "========================================================"

# Show summary of downloaded data
echo "Downloaded data summary:"
find "$OUTPUT_DIR" -name "*.zarr" -type d | head -20
total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
echo "Total size: $total_size"