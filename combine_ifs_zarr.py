#!/usr/bin/env python3
"""
Combine per-date IFS Zarr archives into consolidated, init_time-sorted outputs.

Usage:
    python combine_ifs_zarr.py <output_dir> [--model esfm] [--log-file logs/combine_ifs.log]

The script expects archives at <output_dir>/<YYYYMMDDHHMM>/<model>/(ifs_ens.zarr|ifs_control.zarr)
and writes the combined outputs to:
        <output_dir>/<model>/ifs_ens_combined.zarr
        <output_dir>/<model>/ifs_control_combined.zarr
"""

import argparse
import glob
import logging
import os
from datetime import datetime

import xarray as xr


def setup_logging(log_file: str | None) -> None:
    level = logging.INFO
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=level,
            format=fmt,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=level, format=fmt)


def find_archives(output_dir: str, model: str, name: str) -> list[tuple[datetime, str]]:
    pattern = os.path.join(output_dir, "20*", model, name)
    items: list[tuple[datetime, str]] = []
    for path in glob.glob(pattern):
        try:
            date_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        except Exception:
            continue
        try:
            dt = datetime.strptime(date_dir, "%Y%m%d%H%M")
        except Exception:
            logging.warning("Skipping path with non-conforming date dir: %s", path)
            continue
        items.append((dt, path))
    items.sort(key=lambda t: t[0])
    return items


def combine_and_write(items: list[tuple[datetime, str]], out_path: str, label: str) -> bool:
    if not items:
        logging.info(f"No {label} archives found to combine.")
        return False
    logging.info(f"Combining {len(items)} {label} archives into {out_path}")
    dsets: list[xr.Dataset] = []
    for _, path in items:
        try:
            ds = xr.open_zarr(path, consolidated=True)
            dsets.append(ds)
            logging.info("  included %s: %s", label, path)
        except Exception as e:
            logging.error("Failed to open %s: %s", path, e)
    if not dsets:
        logging.warning(f"No readable {label} datasets; skipping write.")
        return False
    try:
        # Concatenate along init_time (assumed present in per-range archives)
        combined = xr.concat(
            dsets,
            dim="init_time",
            data_vars="minimal",
            coords="minimal",
            compat="no_conflicts",
            join="outer",
        )
        combined = combined.sortby("init_time")

        # Build chunking dict based on available dimensions
        chunk_dict = {
            "init_time": 1,
            "lead_time": 1,
            "latitude": -1,
            "longitude": -1,
        }

        # Only add 'level' if it exists in dimensions
        if "level" in combined.dims:
            chunk_dict["level"] = 1

        # Rechunk the dataset
        logging.info(f"Rechunking with: {chunk_dict}")
        combined = combined.chunk(chunk_dict)

        # Clean up problematic attributes that conflict with encoding
        # Remove 'dtype' from coordinate attributes if present
        for coord_name in combined.coords:
            if "dtype" in combined[coord_name].attrs:
                logging.info(f"Removing 'dtype' attribute from {coord_name}")
                combined[coord_name].attrs.pop("dtype")

        # Explicitly set encoding chunks for each variable to ensure they persist on write
        for var_name in combined.data_vars:
            var = combined[var_name]
            encoding_chunks = []
            for dim in var.dims:
                if dim == "ensemble":
                    encoding_chunks.append(combined.sizes["ensemble"])  # Full ensemble size
                elif dim == "init_time":
                    encoding_chunks.append(1)
                elif dim == "lead_time":
                    encoding_chunks.append(1)
                elif dim == "level":
                    encoding_chunks.append(1)
                elif dim == "latitude":
                    encoding_chunks.append(combined.sizes["latitude"])
                elif dim == "longitude":
                    encoding_chunks.append(combined.sizes["longitude"])
                else:
                    encoding_chunks.append(combined.sizes[dim])

            var.encoding["chunks"] = tuple(encoding_chunks)
            logging.info(f"Set encoding chunks for {var_name}: {tuple(encoding_chunks)}")

        # Write out with Zarr v2
        # Ensure parent directory exists (e.g., <output_dir>/<model>)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            logging.info(f"Overwriting existing combined archive: {out_path}")
            import shutil

            shutil.rmtree(out_path)
        combined.to_zarr(out_path, mode="w", consolidated=True, zarr_format=2)
        logging.info(f"Wrote combined archive: {out_path}")
        return True
    except Exception:
        logging.exception(f"Failed to write combined {label} archive")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine IFS Zarr archives into consolidated outputs"
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory containing per-date archives"
    )
    parser.add_argument(
        "--model", type=str, default="esfm", help="Model directory name under each date"
    )
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    args = parser.parse_args()

    setup_logging(args.log_file)

    output_dir = args.output_dir
    model = args.model

    ens_items = find_archives(output_dir, model, "ifs_ens.zarr")
    ctrl_items = find_archives(output_dir, model, "ifs_control.zarr")

    # Place combined outputs under the model folder
    ens_out = os.path.join(output_dir, model, "ifs_ens_combined.zarr")
    ctrl_out = os.path.join(output_dir, model, "ifs_control_combined.zarr")

    ok1 = combine_and_write(ens_items, ens_out, label="ensemble")
    ok2 = combine_and_write(ctrl_items, ctrl_out, label="control")

    if ok1 or ok2:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
