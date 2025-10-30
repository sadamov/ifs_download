#!/usr/bin/env python3
"""
Simplified validation script to check basic setup for IFS download.
"""

import os
import sys
from datetime import datetime


def load_config():
    """Load simple VAR=VALUE pairs from config.env next to this script."""
    cfg = {}
    cfg_path = os.path.join(os.path.dirname(__file__), "config.env")
    try:
        with open(cfg_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    except Exception:
        pass
    return cfg

def check_environment():
    """Check if required Python packages are available."""
    print("Checking Python environment...")
    
    required_packages = [
        "earthkit.data",
        "xarray", 
        "zarr",
        "dask",
        "netcdf4"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "earthkit.data":
                import earthkit.data as pkg
            elif package == "netcdf4":
                import netCDF4 as pkg
            else:
                pkg = __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("All required packages are available!")
    return True


def check_output_directory(output_dir):
    """Check if output directory is accessible and has enough space."""
    print(f"\nChecking output directory: {output_dir}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Test write access
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        print(f"  ✓ Directory is writable")
        
        # Check available space (if possible)
        try:
            import shutil
            total, used, free = shutil.disk_usage(output_dir)
            free_gb = free // (1024**3)
            print(f"  ✓ Available space: {free_gb} GB")
            
            # Warn if less than 1TB free (each week ~350GB * 8 weeks = ~2.8TB)
            if free_gb < 3000:
                print(f"  ⚠ Warning: May need more space for 8 weeks of data (~3TB recommended)")
        except Exception:
            print("  ? Could not check available space")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Error accessing directory: {e}")
        return False


def validate_date_ranges(cfg):
    """Validate the date ranges from config.env (DATE_RANGES) or defaults."""
    print("\nValidating date ranges...")
    if cfg.get("DATE_RANGES"):
        # config.env uses start|end pairs separated by commas
        date_ranges = [r.replace("|", ":") for r in cfg["DATE_RANGES"].split(",") if r]
    else:
        date_ranges = [
            "2023-01-02T00:2023-01-08T23",
            "2023-04-02T00:2023-04-08T23",
            "2023-07-02T00:2023-07-08T23",
            "2023-10-02T00:2023-10-08T23",
            "2024-01-02T00:2024-01-08T23",
            "2024-04-02T00:2024-04-08T23",
            "2024-07-02T00:2024-07-08T23",
            "2024-10-02T00:2024-10-08T23",
        ]
    
    total_days = 0
    
    for i, date_range in enumerate(date_ranges, 1):
        try:
            start_str, end_str = date_range.split(':')
            start_date = datetime.strptime(start_str, "%Y-%m-%dT%H")
            end_date = datetime.strptime(end_str, "%Y-%m-%dT%H")
            
            days = (end_date - start_date).days + 1
            total_days += days
            
            print(f"  ✓ Range {i}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)")
            
        except Exception as e:
            print(f"  ✗ Range {i}: Invalid format - {e}")
            return False
    
    print(f"\nTotal period: {total_days} days ({total_days/7:.1f} weeks)")
    print(f"Estimated data size: ~{total_days/7 * 350:.0f} GB")
    
    return True


def check_ecmwf_credentials():
    """Check if ECMWF credentials are configured."""
    print("\nChecking ECMWF credentials...")
    
    # Check for .ecmwfapirc file
    ecmwfrc_path = os.path.expanduser("~/.ecmwfapirc")
    if os.path.exists(ecmwfrc_path):
        print(f"  ✓ Found credentials file: {ecmwfrc_path}")
        return True
    
    # Check for environment variables
    if os.getenv("ECMWF_API_KEY") and os.getenv("ECMWF_API_URL"):
        print("  ✓ Found ECMWF environment variables")
        return True
    
    print("  ✗ No ECMWF credentials found")
    print("    Please set up credentials:")
    print("    1. Visit https://api.ecmwf.int/v1/key/ to get your API key")
    print("    2. Create ~/.ecmwfapirc file or set environment variables")
    print("    3. For MARS access, you may need special permissions")
    return False


def main():
    print("IFS Bulk Download - Basic Setup Validation")
    print("="*50)

    cfg = load_config()
    output_dir = cfg.get("OUTPUT_DIR", "/capstor/store/cscs/swissai/a122/IFS")
    
    # Run all checks
    checks = [
        check_environment(),
        check_output_directory(output_dir),
        validate_date_ranges(cfg),
    ]
    
    # Check credentials but don't fail if missing (can be set up later)
    cred_check = check_ecmwf_credentials()
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    if all(checks):
        print("✓ Basic setup checks passed!")
        if cred_check:
            print("✓ ECMWF credentials found")
            print("\nReady to run bulk download!")
        else:
            print("⚠ ECMWF credentials need to be set up")
            print("  You'll need MARS access permissions for the download to work")
        
        print("\nTo start the download, run:")
        print("  sbatch submit_ifs_download.sh")
        
        if cred_check:
            return 0
        else:
            print("\nNote: Set up ECMWF credentials before running the download.")
            return 0
    else:
        print("✗ Some basic checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())