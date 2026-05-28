# IFS Bulk Download Scripts

Streamlined tools to download IFS (Integrated Forecasting System) data from ECMWF/MARS for multiple date ranges, with SLURM job helpers and a tiny debug mode for quick validation.

## Works anywhere

You can clone and run this repository from any directory. Outputs can be written to any filesystem location by setting `OUTPUT_DIR` in `config.env` (absolute or relative). If `OUTPUT_DIR` is not set, the default is a local folder `./ifs_output` under the current working directory.

## Quick start (4 steps)

### 1. Configure once

Edit `config.env` to set your output path, date ranges, and options. Keys (see file for examples and defaults):

- OUTPUT_DIR: output directory for data
- INTERVAL: forecast step in hours
- DOWNLOAD_TYPE: `ensemble` | `control` | `both`
- ENSEMBLE_MARS_CHUNK_SIZE: members per MARS request (1-50, default 50 to minimize catalogue calls)
- DEBUG_SMALL: `0` or `1` (1 uses a tiny subset for fast checks)
- MODEL_NAME: model directory written under each date
- IFS_GRID / IFS_AREA / IFS_PRESSURE_LEVELS / IFS_PRESSURE_LEVEL_PARAMS / IFS_SINGLE_LEVEL_PARAMS: comma-separated geometry and parameter lists for the requests
- ECCODES_DIR: optional path to ecCodes install (or leave empty for auto-detect)
- DATE_RANGES: comma-separated list of `start|end` pairs

Date ranges are constructed from the `DATE_RANGES` setting in `config.env`; you don’t need to edit the scripts for this.

### 2. Navigate to the repository

```bash
cd <path-to-your-cloned-repo>
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

- By default, 8 periods (8 weeks total, 56 days) covering Jan/Apr/Jul/Oct for 2023 and 2024 (controlled by `DATE_RANGES`):
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

## Files included

- Download scripts: `download_ifs_range.py`, `combine_ifs_zarr.py`
- SLURM scripts: `submit_ifs_download.sh`, `submit_ifs_download_chain.sh` (chained submissions)
- IC-only sibling: `download_ic_perturbed.py` + `submit_ic_perturbed.sh` + `config_ic_perturbed.env` (see next section)
- Validation: `validate_env_quick.py`, `validate_env_full.py`
- Virtual environment: `.venv/` with required packages
- Documentation: this `README.md`

## IC-only ensemble download (`download_ic_perturbed.py`)

Sibling script for downloading **only the IFS ENS perturbed analyses** (no
forecast steps), intended as initial conditions for AI weather forecast models
(Aurora, GraphCast, SFNO).

What it downloads per (date, time):
- **Upper-air on pressure levels (PL)**: 13 levels including 600 hPa --
  `1000/925/850/700/600/500/400/300/250/200/150/100/50`. Params: T, U, V, Q, Z, W
  (130/131/132/133/129/135).
- **Surface (SFC)**: MSL, 2T, 10U, 10V, 100U, 100V, SP, TCWV
  (151/167/165/166/246/247/134/137).

MARS request: `class=od stream=enfo type=pf step=0 number=1..50`. `type=pf step=0`
is the actual per-member perturbed analysis (the IC ready to feed a model) --
simpler than `type=icp` which would require re-adding the control analysis per
member.

Init time grid: 8 weeks (Jan / Apr / Jul / Oct in 2023 and 2024), each extended
6 hours backward to include the `T-6h` slot needed by Aurora / GraphCast (both
autoregressive over [t-6h, t]). Generated in code as 28 6-hourly samples per
week × 8 = **224 init times**.

Output: one Zarr v3 sharded store at
`/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr` (default).
Dims: `(ensemble=50, init_time=224, [level=13 for 3D vars], latitude=721,
longitude=1440)`. No `lead_time` dim (this is analysis, not forecast).
Storage: ~3.8 TB.

### Submit

```bash
# Paranoia check first: download ONE init's PL+SFC pair, print summary, no zarr write
sbatch --account=a122 --partition=normal --time=4:00:00 --mem=128G -c 8 \
    --output=logs/paranoia_%j.out --error=logs/paranoia_%j.err \
    --wrap="cd $(pwd) && .venv/bin/python -u download_ic_perturbed.py --single-init"

# Full 224-init download (idempotent: already-written init slices are skipped)
sbatch submit_ic_perturbed.sh
```

### Environment gotchas the submit script handles

- `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` -> certifi's bundle. CSCS compute
  nodes do not ship a system CA bundle, so without this the ECMWF API HTTPS
  connection fails certificate verification.
- `TMPDIR` and `DASK_TEMPORARY_DIRECTORY` -> `/iopsstor/scratch/...`.
  `/tmp` on Clariden compute nodes is too small (~94 GB) for any
  meaningful download or dask spill.
- `PYTHONUNBUFFERED=1` + `python -u` -> stdout flushes line-by-line so
  the SLURM log file shows live progress from the ECMWF API client.

## Prerequisites

- MARS access: ECMWF account with MARS permissions
- Credentials: create `~/.ecmwfapirc` with your API key (get key from <https://api.ecmwf.int/v1/key/>)

The fuller validator `validate_env_full.py` will attempt a tiny ECMWF request via `earthkit.data` to confirm access.

## Monitoring and logs

```bash
# Check job status
squeue -u $USER

# Monitor main job logs
tail -f logs/ifs_download_main_*.out

# (Per-range Python logs were removed for simplicity; use the main job log above.)

# Check downloaded data (replace with your configured output dir)
ls -la "$OUTPUT_DIR"
```

## Detailed usage

### Single date-range download

```bash
python download_ifs_range.py <OUTPUT_DIR> 202301020000 7 --interval 6 --download-type both --range-end 202301082300
```

Very fast test with tiny debug subset:

```bash
python download_ifs_range.py <OUTPUT_DIR> 202301020000 1 --interval 6 --download-type both --debug-small
```

Parameters:

- `output_dir`: Output directory
- `date_time`: Start date in `YYYYMMDDHHMM`
- `num_days`: Number of days to download
- `--interval`: Time step in hours (default: 6)
- `--range-end`: Inclusive end datetime (YYYYMMDDHHMM). When set, the script fetches every init_time between `date_time` and `range-end` in one go (respecting `--interval`).
- `--download-type`: `ensemble`, `control`, or `both` (default: both)
- `--debug-small`: Tiny subset (coarse grid, small area, 1 PL level/var, 1 SL var, steps 0 and 1, 2 ensemble members)

### Customization

Prefer editing `config.env` to change the download campaign. If needed, you can also edit `submit_ifs_download.sh`:

- Date ranges: set `DATE_RANGES` (comma-separated `start|end` pairs)
- Download type: set `DOWNLOAD_TYPE` (`ensemble`, `control`, or `both`)
- Output location: set `OUTPUT_DIR`
- Tiny debug subset: set `DEBUG_SMALL=1` (adds `--debug-small` to the Python call)

You can also pass extra sbatch flags when chaining, after `--` (e.g., partition, time).

## Data structure

```text
<OUTPUT_DIR>/
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

## Field defaults (ESFM)

The Python downloader now reads request geometry and parameter selections from `config.env`:

```dotenv
# 0.25° grid at global coverage
IFS_GRID=0.25,0.25
IFS_AREA=90,-180,-90,180

# Pressure-level setup (ECMWF ENFO PL set omits 600 hPa)
IFS_PRESSURE_LEVELS=1000,925,850,700,500,400,300,250,200,150,100,50
IFS_PRESSURE_LEVEL_PARAMS=u,v,t,q,z

# Surface variables (geopotential removed to avoid duplicate fields)
IFS_SINGLE_LEVEL_PARAMS=2t,10u,10v,msl,tp
```

Edit these entries to tailor the downloads (e.g., set a smaller area or drop certain variables). All values are comma-separated; whitespace is ignored.

## Troubleshooting

Common issues and quick checks:

1. Authentication error — verify `.ecmwfapirc` and network access

```text
Check your ECMWF credentials and network access
```

1. Disk space — ensure capacity on the target filesystem

```bash
df -h "$OUTPUT_DIR"
```

1. Import error — activate the correct environment or install missing packages

```text
ImportError: No module named 'earthkit.data'
```

1. SLURM job failed — check logs in `logs/`, verify account/partition, and resource limits

### Monitoring helpers

```bash
squeue -u $USER
tail -f logs/ifs_download_main_*.out
find "$OUTPUT_DIR" -name "*.zarr" -type d | wc -l
du -sh "$OUTPUT_DIR"
```

## Performance notes

- Each 7-day period typically takes ~12 hours depending on system load
- Full campaign (8 ranges): ~3–5 days
- Ensemble requests default to a single 50-member call; override `ENSEMBLE_MARS_CHUNK_SIZE` if you need smaller batches for memory reasons
- Each configured date range is downloaded via a single Python invocation that requests every init_time within the range, minimizing the number of MARS catalogue lookups.
- Requests are chunked to respect MARS limits; downloads resume and skip existing outputs

## After the download: combine Zarr stores

Combination is executed automatically at the end of `submit_ifs_download.sh`. You can also run it manually:

```bash
python combine_ifs_zarr.py <OUTPUT_DIR> --model esfm
```

Outputs are written under the model folder, for example:

```text
<OUTPUT_DIR>/<MODEL_NAME>/ifs_ens_combined.zarr
<OUTPUT_DIR>/<MODEL_NAME>/ifs_control_combined.zarr
```

Note: `submit_ifs_download.sh` automatically reads `config.env` and passes `MODEL_NAME` to the combine step. If you set a different model name in `config.env`, use that here instead of `esfm`.

## Support

1. Run `.venv/bin/python validate_env_full.py` to check your setup (and optionally MARS access)
2. Inspect logs in the `logs/` directory
3. Verify ECMWF account and MARS permissions
4. Ensure sufficient disk space is available

## Data and Terms

This project only provides code. Any ECMWF, Copernicus, or other third‑party data you download with it remain subject to their respective licences and terms.

- ECMWF Open Data Licence: [https://www.ecmwf.int/en/forecasts/datasets/open-data-licence](https://www.ecmwf.int/en/forecasts/datasets/open-data-licence)
- Copernicus Climate Data Store Terms: [https://cds.climate.copernicus.eu/terms](https://cds.climate.copernicus.eu/terms)
- Check any institutional/archive-specific terms that may apply.

Users are responsible for ensuring compliance when redistributing datasets produced by this code.
