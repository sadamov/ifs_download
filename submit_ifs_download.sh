#!/bin/bash

#SBATCH --job-name=ifs_download_main          # short name for the job
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH -c 72                       # CPU cores per task
#SBATCH --mem=460000                # memory per node
#SBATCH --exclusive
#SBATCH --time=12:00:00             # total run time (HH:MM:SS)
#SBATCH --account=a122
#SBATCH --partition=normal             # partition name
#SBATCH --output=logs/ifs_download_main_%j.out  # output log file
#SBATCH --requeue

# Configuration via config.env (single source of truth)
# You can override the config file path by exporting CONFIG_FILE before sbatch
#   e.g., CONFIG_FILE=/path/to/config.env sbatch submit_ifs_download.sh
CONFIG_FILE="${CONFIG_FILE:-$(pwd)/config.env}"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    # Read only KEY=VALUE lines and assign/export safely without evaluating special characters
    # This avoids issues with characters like '|' being treated as pipes when sourcing.
    while IFS= read -r __line; do
        __key="${__line%%=*}"
        __val="${__line#*=}"
        # Assign literal value and export
        printf -v "${__key}" "%s" "${__val}"
        export "${__key}"
    done < <(
        sed -e 's/\r$//' -n \
            -e '/^[[:space:]]*#/{d;}' \
            -e '/^[[:space:]]*$/d' \
            -e '/^[A-Za-z_][A-Za-z0-9_]*=.*/p' -- "$CONFIG_FILE"
    )
else
    echo "No config file found at $CONFIG_FILE, using built-in defaults"
fi

# Defaults if not provided by config
# By default, store outputs under the current working directory
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/ifs_output}"
export INTERVAL="${INTERVAL:-6}"
export DOWNLOAD_TYPE="${DOWNLOAD_TYPE:-both}"
export DEBUG_SMALL="${DEBUG_SMALL:-0}"
export MODEL_NAME="${MODEL_NAME:-esfm}"

# ecCodes library setup (required by earthkit/eccodes)
# Try to auto-detect an existing ecCodes installation (e.g., in Miniforge/Conda 'apps' env)
if [ -z "${ECCODES_DIR:-}" ]; then
    for CAND in \
        "$HOME/miniforge3/envs/apps" \
        "$HOME/miniconda3/envs/apps" \
        "/users/$USER/miniforge3/envs/apps" \
        "/users/$USER/miniconda3/envs/apps"; do
        if [ -f "$CAND/lib/libeccodes.so" ]; then
            export ECCODES_DIR="$CAND"
            break
        fi
    done
fi

if [ -n "${ECCODES_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${ECCODES_DIR}/lib:${LD_LIBRARY_PATH}"
    if [ -d "${ECCODES_DIR}/share/eccodes/definitions" ]; then
        export ECCODES_DEFINITION_PATH="${ECCODES_DIR}/share/eccodes/definitions"
    fi
    if [ -d "${ECCODES_DIR}/share/eccodes/samples" ]; then
        export ECCODES_SAMPLES_PATH="${ECCODES_DIR}/share/eccodes/samples"
    fi
fi

# Create logs directory
mkdir -p logs



echo "========================================================"
echo "Starting IFS bulk download"
echo "========================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Download type: $DOWNLOAD_TYPE"
echo "Debug small: $DEBUG_SMALL"
echo "Start time: $(date)"
echo "========================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Prefer repository virtualenv python if available
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Define the date ranges and process each one
# Use '|' delimiter between start and end; ranges provided via DATE_RANGES in config.env
declare -a date_ranges=()
if [ -n "${DATE_RANGES:-}" ]; then
    IFS=',' read -r -a date_ranges <<< "${DATE_RANGES}"
else
    date_ranges=(
        "2023-01-02T00|2023-01-08T23"
        "2023-04-02T00|2023-04-08T23"
        "2023-07-02T00|2023-07-08T23"
        "2023-10-02T00|2023-10-08T23"
        "2024-01-02T00|2024-01-08T23"
        "2024-04-02T00|2024-04-08T23"
        "2024-07-02T00|2024-07-08T23"
        "2024-10-02T00|2024-10-08T23"
    )
fi

# In debug-small mode, drastically reduce to two very short windows
if [ "$DEBUG_SMALL" = "1" ] || [ "$DEBUG_SMALL" = "true" ] || [ "$DEBUG_SMALL" = "TRUE" ]; then
    date_ranges=(
        "2023-01-02T00|2023-01-02T00"
        "2024-01-02T00|2024-01-02T00"
    )
fi

# Process each date range
total_ranges=${#date_ranges[@]}
success_count=0
failed_ranges=()

MAX_LEAD_HOURS=${MAX_LEAD_TIME_HOURS:-240}
forecast_days=$(( (MAX_LEAD_HOURS + 23) / 24 ))
if [ "$forecast_days" -lt 1 ]; then
    forecast_days=1
fi
init_step_hours=${INIT_INTERVAL_HOURS:-$INTERVAL}
init_step_seconds=$((init_step_hours * 3600))

for i in "${!date_ranges[@]}"; do
    range_num=$((i + 1))
    date_range="${date_ranges[$i]}"

    echo ""
    echo "========================================================"
    echo "Processing range $range_num/$total_ranges: $date_range"
    echo "========================================================"

    IFS='|' read -r start_datetime end_datetime <<< "$date_range"
    start_iso="${start_datetime}:00:00"
    end_iso="${end_datetime}:00:00"

    start_label=$(date -u -d "$start_iso" +%Y%m%d%H%M)
    end_label=$(date -u -d "$end_iso" +%Y%m%d%H%M)
    start_human=$(date -u -d "$start_iso" +"%Y-%m-%d %H:%M UTC")
    end_human=$(date -u -d "$end_iso" +"%Y-%m-%d %H:%M UTC")

    echo "Init span: $start_human -> $end_human (interval ${INTERVAL}h)"
    echo "Forecast days per init: $forecast_days (derived from MAX_LEAD_TIME_HOURS=${MAX_LEAD_HOURS})"

    DEBUG_ARG=""
    if [ "$DEBUG_SMALL" = "1" ] || [ "$DEBUG_SMALL" = "true" ] || [ "$DEBUG_SMALL" = "TRUE" ]; then
        DEBUG_ARG="--debug-small"
    fi

    if "$PYTHON_BIN" download_ifs_range.py "$OUTPUT_DIR" "$start_label" "$forecast_days" --interval "$INTERVAL" --download-type "$DOWNLOAD_TYPE" --range-end "$end_label" $DEBUG_ARG; then
        echo "✓ Successfully completed range $range_num/$total_ranges"
        ((success_count++))
    else
        echo "✗ Range $range_num/$total_ranges failed"
        failed_ranges+=("$date_range")
    fi

    echo "Progress: $success_count/$range_num ranges completed successfully"
done

echo ""
echo "========================================================"
echo "BULK DOWNLOAD SUMMARY"
echo "========================================================"
echo "Total ranges processed: $total_ranges"
echo "Successful downloads: $success_count"
echo "Failed downloads: $((total_ranges - success_count))"
echo "End time: $(date)"

if [ ${#failed_ranges[@]} -gt 0 ]; then
    echo ""
    echo "Failed ranges:"
    for failed_range in "${failed_ranges[@]}"; do
        echo "  - $failed_range"
    done
fi

echo ""
echo "Data summary:"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
    echo "Number of date directories: $(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "20*" | wc -l)"
    echo "Number of zarr files: $(find "$OUTPUT_DIR" -name "*.zarr" -type d | wc -l)"
    total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
    echo "Total size: $total_size"
else
    echo "Output directory not found"
fi

echo "========================================================"

# Combine per-range archives into consolidated Zarr stores (Zarr v2)
echo "Combining per-range archives into consolidated outputs..."
if "$PYTHON_BIN" combine_ifs_zarr.py "$OUTPUT_DIR" --model "$MODEL_NAME"; then
    echo "✓ Combination complete. Combined outputs (if present):"
    [ -d "$OUTPUT_DIR/$MODEL_NAME/ifs_ens_combined.zarr" ] && echo "  - $OUTPUT_DIR/$MODEL_NAME/ifs_ens_combined.zarr"
    [ -d "$OUTPUT_DIR/$MODEL_NAME/ifs_control_combined.zarr" ] && echo "  - $OUTPUT_DIR/$MODEL_NAME/ifs_control_combined.zarr"
else
    echo "✗ Combination step failed. See main job log for details."
fi

# Exit with error code if any downloads failed
if [ $success_count -eq $total_ranges ]; then
    echo "All downloads completed successfully!"
    exit 0
else
    echo "Some downloads failed. Check logs for details."
    exit 1
fi
