#!/bin/bash

#SBATCH --job-name=ifs_combine
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --account=a122
#SBATCH --partition=normal
#SBATCH --output=logs/ifs_combine_%j.out

set -euo pipefail

# Load configuration
CONFIG_FILE="${CONFIG_FILE:-$(pwd)/config.env}"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    while IFS= read -r __line; do
        __key="${__line%%=*}"
        __val="${__line#*=}"
        printf -v "${__key}" "%s" "${__val}"
        export "${__key}"
    done < <(
        sed -e 's/\r$//' -n \
            -e '/^[[:space:]]*#/{d;}' \
            -e '/^[[:space:]]*$/d' \
            -e '/^[A-Za-z_][A-Za-z0-9_]*=.*/p' -- "$CONFIG_FILE"
    )
fi

# Defaults
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/ifs_output}"
export MODEL_NAME="${MODEL_NAME:-esfm}"

# Python setup
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "========================================================"
echo "Starting IFS Zarr Combination"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "========================================================"

"$PYTHON_BIN" combine_ifs_zarr.py "$OUTPUT_DIR" --model "$MODEL_NAME"

echo "Done."
