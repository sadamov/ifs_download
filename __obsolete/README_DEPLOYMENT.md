# IFS Download Repository - Deployment Copy

This repository has been cleaned and prepared for deployment to `/capstor/store/cscs/swissai/a122/IFS/repo-download-ifs`.

## What was cleaned/removed:
- Git history and sensitive information
- User-specific paths updated to new location
- Temporary files and cache
- User-specific credentials (if any)

## Setup for colleagues:

### 1. Test the environment:
```bash
cd /capstor/store/cscs/swissai/a122/IFS/repo-download-ifs
.venv/bin/python validate_setup_simple.py
```

### 2. Set up ECMWF credentials:
You'll need to set up your own ECMWF credentials with MARS access:
```bash
# Create ~/.ecmwfapirc with your credentials
# Visit: https://api.ecmwf.int/v1/key/
```

### 3. Run the download:
```bash
# For SLURM submission:
sbatch submit_ifs_bulk_master.sh

# Or modify the date ranges and parameters as needed
```

## Important Notes:
- You need MARS access permissions from ECMWF
- The virtual environment is pre-configured with all required packages
- Output directory is set to `/capstor/store/cscs/swissai/a122/IFS`
- All absolute paths have been updated to the new location

## Files included:
- IFS download scripts (both ensemble and control)
- SLURM batch submission scripts
- Validation scripts
- Pre-configured Python virtual environment
- Documentation

For questions, refer to the main README_IFS_DOWNLOAD.md file.
