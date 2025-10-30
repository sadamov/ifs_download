# 🌟 IFS Data Download Instructions for Colleagues

## 📍 **Repository Location**

```text
/capstor/store/cscs/swissai/a122/IFS/repo-download-ifs
```

## 🚀 **Quick Start (3 Steps)**

### 1️⃣ **Navigate to the repository:**

```bash
cd /capstor/store/cscs/swissai/a122/IFS/repo-download-ifs
```

### 2️⃣ **Validate your setup:**

```bash
.venv/bin/python validate_setup_simple.py
```

If you want the validator to also test MARS/ECMWF access (this will perform a small metadata request and requires a valid `~/.ecmwfapirc` with MARS permissions), run the fuller check before submitting the job:

```bash
.venv/bin/python validate_setup.py
```

### 3️⃣ **Submit the download job:**

Option A — single run (will stop at time limit):

```bash
sbatch submit_ifs_bulk_master.sh
```

Option B — recommended for long runs: queue a chain of jobs so the next starts when the previous ends (including TIMEOUT):

```bash
# Queue 4 chained runs; next starts after the previous finishes for any reason
./submit_ifs_bulk_chain.sh -n 4

# Only continue when the previous did NOT complete OK (captures TIMEOUT/FAILED)
./submit_ifs_bulk_chain.sh -n 6 -d afternotok

# Pass extra sbatch options after -- (override partition, time, etc.)
./submit_ifs_bulk_chain.sh -n 3 -- --partition=normal --time=12:00:00
```

## ⚠️ **Important Prerequisites**

- **MARS Access Required**: You need ECMWF credentials with MARS access permissions
- **Set up credentials**: Create `~/.ecmwfapirc` file with your ECMWF API key
- **Get API key from**: <https://api.ecmwf.int/v1/key/>

### MARS/ECMWF access check

The included validator `validate_setup.py` performs a quick MARS/ECMWF access test using `earthkit.data` (it tries a small metadata request). Make sure:

- You have a valid `~/.ecmwfapirc` file with your API key.
- Your account has MARS access permissions.

If the validator reports "MARS access failed", check your `~/.ecmwfapirc`, network access, and that your ECMWF account has MARS permissions.

## 📊 **What You'll Get**

- **8 time periods** (8 weeks total, 56 days)
- **Seasons covered**: January, April, July, October for 2023 and 2024
- **Data types**: Both ensemble (50 members) and control forecasts
- **Format**: Zarr files (efficient for analysis)
- **Size**: ~2.8 TB total
- **Location**: Data will be saved to `/capstor/store/cscs/swissai/a122/IFS/`

## 🔧 **Customization Options**

Edit `submit_ifs_bulk_master.sh` to modify:

- **Date ranges**: Update the `date_ranges` array
- **Download type**: Set `DOWNLOAD_TYPE` (ensemble, control, or both)
- **Output location**: Modify `OUTPUT_DIR`
- **Tiny debug subset**: set `DEBUG_SMALL=1` when submitting to download a very small subset (small area, coarse grid, steps 0 and 1, 2 ensemble members)

## 📝 **Monitoring Progress**

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

## 🛠️ **Files Included**

- ✅ Download scripts: `download_ifs_single.py`, `combine_ifs_zarr.py`
- ✅ SLURM scripts: `submit_ifs_bulk_master.sh`, `submit_ifs_bulk_chain.sh` (new: chained submissions)
- ✅ Validation: `validate_setup_simple.py`, `validate_setup.py`
- ✅ Virtual environment: `.venv/` with all required packages
- ✅ Documentation: `README_IFS_DOWNLOAD.md`, this `QUICK_START.md`

Note: older bulk scripts live under `__obsolete/` and are not used in the current flow.

## 🆘 **Troubleshooting**

| Problem | Solution |
|---------|----------|
| "No MARS access" | Contact ECMWF to request MARS permissions |
| "Package missing" | Virtual environment is pre-configured |
| "Disk space" | ~3TB needed, check `df -h /capstor/store/cscs/swissai/a122/IFS` |
| "Job fails" | Check logs in `logs/` directory |

## 💡 **Tips**

- The download takes 16-48 hours depending on system load
- Scripts automatically skip existing data (resumable)
- Each week is ~350GB, so start with a subset if testing
- All paths have been updated to work in the new location

---
**Ready to download weather data for your research!** 🌦️📈
