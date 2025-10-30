# IFS Bulk Download Scripts

Streamlined tools to download IFS (Integrated Forecasting System) data from ECMWF/MARS for multiple date ranges, with SLURM job helpers and a tiny debug mode for quick validation.

This README consolidates the former QUICK_START and README_IFS_DOWNLOAD documents.

## Repository location (on CSCS)

```text
/capstor/store/cscs/swissai/a122/IFS/repo-download-ifs
```

## Quick start (3 steps)

### 1. Configure once

Edit `config.env` to set your output path, date ranges, and options. Keys:

- OUTPUT_DIR: output directory (default: `/capstor/store/cscs/swissai/a122/IFS`)
- INTERVAL: forecast step in hours (default: 6)
- DOWNLOAD_TYPE: `ensemble` | `control` | `both` (default: `both`)
- DEBUG_SMALL: `0` or `1` (1 uses a tiny subset for fast checks)
- MODEL_NAME: model directory under each date (default: `esfm`)
- ECCODES_DIR: optional path to ecCodes install (leave empty to auto-detect)
- DATE_RANGES: comma-separated list of `start|end` pairs, for example:

```text
DATE_RANGES=2023-01-02T00|2023-01-08T23,2023-04-02T00|2023-04-08T23
```

### 2. Navigate to the repository

```bash
cd /capstor/store/cscs/swissai/a122/IFS/repo-download-ifs
```

### 3. Validate your setup

Use the simple validator (fast, no ECMWF call):

```bash
.venv/bin/python validate_env_quick.py
```

Optionally run the fuller validator that performs a tiny MARS metadata request (requires a valid `~/.ecmwfapirc` and MARS permissions):

```bash
.venv/bin/python validate_env_full.py
```

### 4. Submit the download job

Option A — single run (will stop at time limit):

```bash
sbatch submit_ifs_download.sh
```

Option B — recommended for long runs: queue a chain of jobs so the next starts when the previous ends (including TIMEOUT):

```bash
./submit_ifs_download_chain.sh -n 4
./submit_ifs_download_chain.sh -n 6 -d afternotok
./submit_ifs_download_chain.sh -n 3 -- --partition=normal --time=12:00:00
```

## What you’ll get

- 8 periods (8 weeks total, 56 days) covering Jan/Apr/Jul/Oct for 2023 and 2024:
  - 2023-01-02 to 2023-01-08
  - 2023-04-02 to 2023-04-08
  - 2023-07-02 to 2023-07-08
  - 2023-10-02 to 2023-10-08
  - 2024-01-02 to 2024-01-08
  - 2024-04-02 to 2024-04-08
  - 2024-07-02 to 2024-07-08
  - 2024-10-02 to 2024-10-08
- Both ensemble (50 members) and control forecasts
- Zarr outputs suitable for analysis
- Estimated size: ~2.8 TB total
- Output root: `/capstor/store/cscs/swissai/a122/IFS/`

## Files included

- Download scripts: `download_ifs_range.py`, `combine_ifs_zarr.py`
- SLURM scripts: `submit_ifs_download.sh`, `submit_ifs_download_chain.sh` (chained submissions)
- Validation: `validate_env_quick.py`, `validate_env_full.py`
- Virtual environment: `.venv/` with required packages
- Documentation: this `README.md`

## Prerequisites

- MARS access: ECMWF account with MARS permissions
- Credentials: create `~/.ecmwfapirc` with your API key (get key from <https://api.ecmwf.int/v1/key/>)

The fuller validator `validate_setup.py` will attempt a tiny ECMWF request via `earthkit.data` to confirm access.

## Monitoring and logs

```bash
# Check job status
squeue -u $USER

# Monitor master job logs
tail -f logs/ifs_bulk_master_*.out

# Monitor per-range Python logs
tail -f logs/ifs_download_*.log

# Check downloaded data
ls -la /capstor/store/cscs/swissai/a122/IFS/
```

## Detailed usage

### Single date-range download

```bash
python download_ifs_range.py /capstor/store/cscs/swissai/a122/IFS 202301020000 7 --interval 6 --download-type both
```

Very fast test with tiny debug subset:

```bash
python download_ifs_range.py /capstor/store/cscs/swissai/a122/IFS 202301020000 1 --interval 6 --download-type both --debug-small
```

Parameters:

- `output_dir`: Output directory
- `date_time`: Start date in `YYYYMMDDHHMM`
- `num_days`: Number of days to download
- `--interval`: Time step in hours (default: 6)
- `--download-type`: `ensemble`, `control`, or `both` (default: both)
- `--debug-small`: Tiny subset (coarse grid, small area, 1 PL level/var, 1 SL var, steps 0 and 1, 2 ensemble members)

### Customization

Edit `submit_ifs_bulk_master.sh` to change the download campaign:

- Date ranges: update the `date_ranges` array
- Download type: set `DOWNLOAD_TYPE` (`ensemble`, `control`, or `both`)
- Output location: modify `OUTPUT_DIR`
- Tiny debug subset: set `DEBUG_SMALL=1` (adds `--debug-small` to the Python call)

You can also pass extra sbatch flags when chaining, after `--` (e.g., partition, time).

## Data structure

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

### Python packages

- `earthkit-data`
- `xarray`
- `zarr`
- `dask`
- `netcdf4`

### Environment options

Use the provided virtual environment:

```bash
.venv/bin/python validate_env_quick.py
```

Or create your own conda environment:

```bash
conda create -n ifs-downloads python=3.10 -y
conda activate ifs-downloads
pip install earthkit-data xarray zarr dask netcdf4
```

### System requirements

- ~3TB free disk space
- Access to a SLURM batch system
- Network access to ECMWF/MARS

## Defaults (ESFM)

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

Common issues and quick checks:

1. Authentication error — verify `.ecmwfapirc` and network access

```text
Check your ECMWF credentials and network access
```

1. Disk space — ensure capacity on the target filesystem

```bash
df -h /capstor/store/cscs/swissai/a122/IFS
```

1. Import error — activate the correct environment or install missing packages

```text
ImportError: No module named 'earthkit.data'
```

1. SLURM job failed — check logs in `logs/`, verify account/partition, and resource limits

### Monitoring helpers

```bash
squeue -u $USER
tail -f logs/ifs_bulk_master_*.out
find /capstor/store/cscs/swissai/a122/IFS -name "*.zarr" -type d | wc -l
du -sh /capstor/store/cscs/swissai/a122/IFS
```

## Performance notes

- Each 7-day period typically takes ~12 hours depending on system load
- Full campaign (8 ranges): ~3–5 days
- Ensemble is fetched in 10-member chunks (5 chunks total for 50 members)
- Requests are chunked to respect MARS limits; downloads resume and skip existing outputs

## After the download: combine Zarr stores

Combination is executed automatically at the end of `submit_ifs_download.sh`. You can also run it manually:

```bash
python combine_ifs_zarr.py /capstor/store/cscs/swissai/a122/IFS --model esfm --log-file logs/combine_ifs_manual.log

Note: `submit_ifs_download.sh` automatically reads `config.env` and passes `MODEL_NAME` to the combine step. If you set a different model name in `config.env`, use that here instead of `esfm`.
```

## Support

1. Run `.venv/bin/python validate_setup.py` to check your setup (and optionally MARS access)
2. Inspect logs in the `logs/` directory
3. Verify ECMWF account and MARS permissions
4. Ensure sufficient disk space is available
