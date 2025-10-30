# IFS Bulk Download Scripts (ESFM-only)

This repository contains scripts to download IFS (Integrated Forecasting System) data from ECMWF/MARS for multiple date ranges. The implementation is streamlined for the ESFM configuration and supports a tiny-subset debug mode for quick validation.

## Overview

The scripts will download both ensemble and control IFS data for 8 periods (8 weeks total):

- 2023-01-02 to 2023-01-08 (7 days)
- 2023-04-02 to 2023-04-08 (7 days)
- 2023-07-02 to 2023-07-08 (7 days)
- 2023-10-02 to 2023-10-08 (7 days)
- 2024-01-02 to 2024-01-08 (7 days)
- 2024-04-02 to 2024-04-08 (7 days)
- 2024-07-02 to 2024-07-08 (7 days)
- 2024-10-02 to 2024-10-08 (7 days)

**Total**: 56 days of data, estimated size: ~2.8 TB

**Output directory**: `/capstor/store/cscs/swissai/a122/IFS`

## Files Created

1. **`download_ifs_single.py`** — Single-range downloader (ESFM-only) with error handling and debug mode
2. **`submit_ifs_bulk_master.sh`** — SLURM batch script that iterates over predefined date ranges
3. **`submit_ifs_bulk_chain.sh`** — Helper to submit multiple `submit_ifs_bulk_master.sh` jobs in a dependency chain
4. **`combine_ifs_zarr.py`** — Combine per-range Zarr stores into consolidated outputs
5. **`validate_setup.py` / `validate_setup_simple.py`** — Setup validation scripts
6. **`README_IFS_DOWNLOAD.md`** — This documentation

Legacy scripts (original bulk implementation) are retained under `__obsolete/` and are not used in the current flow.

## Quick Start

### 1. Validate Setup

```bash
# Run validation (checks Python deps and optionally MARS access)
python validate_setup.py
```

### 2. Submit the Job

```bash
# Submit the bulk download job (normal/full data)
sbatch submit_ifs_bulk_master.sh

# Submit the bulk download job in tiny debug mode (fast validation)
DEBUG_SMALL=1 sbatch submit_ifs_bulk_master.sh
```

### 2b. Chained submissions (recommended for long runs)

Avoid manual requeueing and traps by pre-queuing a chain of dependent jobs. The next job starts when the previous finishes (including TIMEOUT):

```bash
# Start 4 chained jobs; each starts after the previous ends (any result)
./submit_ifs_bulk_chain.sh -n 10

# Start 6 chained jobs, continuing only when the previous did NOT finish OK (captures FAILED/TIMEOUT)
./submit_ifs_bulk_chain.sh -n 6 -d afternotok

# Pass extra sbatch args after -- to override headers
./submit_ifs_bulk_chain.sh -n 3 -- --partition=normal --time=12:00:00
```

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Monitor master job logs
tail -f logs/ifs_bulk_master_*.out

# Monitor per-range Python logs
tail -f logs/ifs_download_*.log

# Check partial results
ls -la /capstor/store/cscs/swissai/a122/IFS/*/esfm/
```

## Detailed Usage

### Individual Date Download

To download data for a single date range (ESFM-only):

```bash
python download_ifs_single.py \
   /capstor/store/cscs/swissai/a122/IFS \
   202301020000 \
   7 \
   --interval 6 \
   --download-type both
```

For a very fast test, add the tiny debug subset:

```bash
python download_ifs_single.py \
   /capstor/store/cscs/swissai/a122/IFS \
   202301020000 \
   1 \
   --interval 6 \
   --download-type both \
   --debug-small
```

Parameters:

- `output_dir`: Output directory
- `date_time`: Start date in YYYYMMDDHHMM format
- `num_days`: Number of days to download
- `--interval`: Time step in hours (default: 6)
- `--download-type`: ensemble, control, or both (default: both)
- `--debug-small`: Tiny subset (coarse grid, small area, 1 PL level/var, 1 SL var, steps 0 and 1, 2 ensemble members)

### Customization

#### Change Output Directory

Edit the `OUTPUT_DIR` variable in the batch scripts:

```bash
export OUTPUT_DIR="/your/custom/path"
```

#### Enable Tiny Debug Subset

Enable in the bulk master submission by setting an environment variable:

```bash
DEBUG_SMALL=1 sbatch submit_ifs_bulk_master.sh
```

This passes `--debug-small` to the Python downloader.

#### Change Date Ranges

Edit the `date_ranges` array in `submit_ifs_bulk_master.sh`:

```bash
declare -a date_ranges=(
   "2023-01-02T00|2023-01-08T23"
   "2023-02-01T00|2023-02-07T23"
    # Add your custom ranges here
)
```

#### Download Only Ensemble or Control

```bash
export DOWNLOAD_TYPE="ensemble"  # or "control" or "both"
```

## Data Structure

The downloaded data will be organized as:

```text
/capstor/store/cscs/swissai/a122/IFS/
├── 202301020000/
│   └── esfm/
│       ├── fields.txt
│       ├── ifs_ens.zarr/
│       └── ifs_control.zarr/
├── 202304020000/
│   └── esfm/
│       ├── fields.txt
│       ├── ifs_ens.zarr/
│       └── ifs_control.zarr/
└── ...
```

## Requirements

### Python Packages

- `earthkit-data`
- `xarray`
- `zarr`
- `dask`
- `netcdf4`

Install with your environment manager of choice (conda, mamba, etc.) and pip as needed, for example:

```bash
conda create -n ifs-downloads python=3.10 -y
conda activate ifs-downloads
pip install earthkit-data xarray zarr dask netcdf4
```

### ECMWF Access

- Valid ECMWF account with MARS access
- Proper authentication setup (`.ecmwfapirc` file or other)

### System Requirements

- ~3TB free disk space
- Access to SLURM batch system
- Network access to ECMWF/MARS

## Default Parameters

The downloader uses ESFM defaults (can be customized in code):

```python
{
   "grid": [0.25, 0.25],            # 0.25° resolution
   "area": [90, -180, -90, 180],    # Global coverage
   "pressure_levels": [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
   "pressure_level_params": ["u", "v", "t", "q", "z"],  # Wind, temp, humidity, geopotential
   "single_level_params": ["2t", "10u", "10v", "msl", "tp", "z"]  # Surface variables
}
```

## Troubleshooting

### Common Issues

1. **Authentication Error**

   ```text
   Check your ECMWF credentials and network access
   ```

   - Verify `.ecmwfapirc` file is properly configured
   - Test with a simple earthkit request

2. **Disk Space Error**

   ```text
   No space left on device
   ```

   - Check available space: `df -h /capstor/store/cscs/swissai/a122/IFS`
   - Clean up old data or use a different output directory

3. **Import Error**

   ```text
   ImportError: No module named 'earthkit.data'
   ```

   - Activate the correct conda environment
   - Install missing packages: `pip install earthkit-data`

4. **SLURM Job Failed**
   - Check logs in `logs/` directory
   - Verify SLURM account and partition settings
   - Check resource limits (memory, time)

### Monitoring

```bash
# Check job status
squeue -u $USER

# View recent log output
tail -f logs/ifs_bulk_master_*.out

# Check downloaded data
find /capstor/store/cscs/swissai/a122/IFS -name "*.zarr" -type d | wc -l

# Check total size
du -sh /capstor/store/cscs/swissai/a122/IFS
```

### Restarting Failed Downloads

The scripts check for existing data and skip downloads if the output already exists. To restart:

1. **Partial restart**: Delete only the failed date directories
2. **Full restart**: Delete entire output directory
3. **Resume**: Just resubmit the job - it will skip completed downloads

## Performance Notes

- Each period (7 days) takes approximately 2-6 hours depending on system load
- Total download time: 16-48 hours for all 8 periods
- Data is downloaded in chunks to avoid MARS request size limits
- Ensemble data is downloaded in 10-member chunks (5 chunks total for 50 members)

## Support

If you encounter issues:

1. Run `python validate_setup.py` to check your setup
2. Check the log files in the `logs/` directory
3. Verify your ECMWF account and MARS access
4. Ensure sufficient disk space is available

For questions about specific parameters, refer to comments and constants in `download_ifs_single.py`.
