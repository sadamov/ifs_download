#!/usr/bin/env python3
"""
Validation script to check setup before running bulk IFS download.
"""

import os
import sys
from datetime import datetime


def check_environment():
    """Check if required Python packages are available."""
    print("Checking Python environment...")

    required_packages = ["earthkit.data", "xarray", "zarr", "dask", "netcdf4"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
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
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

        print("  ✓ Directory is writable")

        # Check available space (if possible)
        try:
            import shutil

            total, used, free = shutil.disk_usage(output_dir)
            free_gb = free // (1024**3)
            print(f"  ✓ Available space: {free_gb} GB")

            # Warn if less than 1TB free (each week ~350GB * 8 weeks = ~2.8TB)
            if free_gb < 3000:
                print("  ⚠ Warning: May need more space for 8 weeks of data (~3TB recommended)")
        except Exception:
            print("  ? Could not check available space")

        return True

    except Exception as e:
        print(f"  ✗ Error accessing directory: {e}")
        return False


def validate_date_ranges():
    """Validate the date ranges."""
    print("\nValidating date ranges...")

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
            start_str, end_str = date_range.split(":")
            start_date = datetime.strptime(start_str, "%Y-%m-%dT%H")
            end_date = datetime.strptime(end_str, "%Y-%m-%dT%H")

            days = (end_date - start_date).days + 1
            total_days += days

            print(
                f"  ✓ Range {i}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)"
            )

        except Exception as e:
            print(f"  ✗ Range {i}: Invalid format - {e}")
            return False

    print(f"\nTotal period: {total_days} days ({total_days / 7:.1f} weeks)")
    print(f"Estimated data size: ~{total_days / 7 * 350:.0f} GB")

    return True


def check_earthkit_access():
    """Test MARS/ECMWF access through earthkit."""
    print("\nTesting MARS/ECMWF access...")

    try:
        import earthkit.data
        from earthkit.data import settings

        # Set a temporary cache directory
        temp_cache = "/tmp/earthkit_test_cache"
        os.makedirs(temp_cache, exist_ok=True)
        settings.set("user-cache-directory", temp_cache)

        # Try a simple request to test access
        print("  Testing connection to MARS...")

        # Small test request - just metadata
        test_request = {
            "class": "od",
            "date": "2023-01-01",
            "expver": "1",
            "levtype": "sfc",
            "param": "2t",
            "step": "0",
            "stream": "oper",
            "time": "00",
            "type": "fc",
            "area": [50, 0, 40, 10],  # Small area
            "grid": [1, 1],  # Coarse grid
        }

        # This will test authentication but not download large data
        ds = earthkit.data.from_source("mars", test_request, lazily=True)
        metadata = ds.metadata()

        print(f"  ✓ MARS access successful (found {len(metadata)} records)")

        # Clean up test cache
        import shutil

        shutil.rmtree(temp_cache, ignore_errors=True)

        return True

    except ImportError:
        print("  ✗ earthkit.data not available")
        return False
    except Exception as e:
        print(f"  ✗ MARS access failed: {e}")
        print("    Check your ECMWF credentials and network access")
        return False


def main():
    print("IFS Bulk Download - Setup Validation")
    print("=" * 50)

    output_dir = "/capstor/store/cscs/swissai/a122/IFS"

    # Run all checks
    checks = [
        check_environment(),
        check_output_directory(output_dir),
        validate_date_ranges(),
        check_earthkit_access(),
    ]

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    if all(checks):
        print("✓ All checks passed! Ready to run bulk download.")
        print("\nTo start the download, run:")
        print("  sbatch submit_ifs_bulk_master.sh")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
