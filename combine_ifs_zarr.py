#!/usr/bin/env python3
"""
Combine per-date IFS Zarr archives into consolidated archives (ensemble and control).
- Ensures Zarr v2 metadata on write
- Combines across the 'time' dimension; if missing, injects a single 'time' coord

Usage:
  python combine_ifs_zarr.py <output_dir> [--model esfm] [--log-file logs/combine_ifs.log]

This script finds archives at: <output_dir>/<YYYYMMDDHHMM>/<model>/(ifs_ens.zarr|ifs_control.zarr)
and writes combined archives to:
    <output_dir>/<model>/ifs_ens_combined.zarr
    <output_dir>/<model>/ifs_control_combined.zarr
"""

import argparse
import glob
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import xarray as xr


def setup_logging(log_file: str | None) -> None:
    level = logging.INFO
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=level,
            format=fmt,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=level, format=fmt)


def find_archives(output_dir: str, model: str, name: str) -> List[Tuple[datetime, str]]:
    pattern = os.path.join(output_dir, "20*", model, name)
    paths = glob.glob(pattern)
    items: List[Tuple[datetime, str]] = []
    for p in paths:
        # Extract the date directory name
        try:
            date_dir = os.path.basename(os.path.dirname(p))  # model dir
            date_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))  # date dir
        except Exception:
            continue
        try:
            dt = datetime.strptime(date_dir, "%Y%m%d%H%M")
        except Exception:
            logging.warning(f"Skipping path with non-conforming date dir: {p}")
            continue
        items.append((dt, p))
    items.sort(key=lambda t: t[0])
    return items


def open_and_tag(path: str, dt: datetime) -> xr.Dataset:
    # Open the zarr archive - time coordinates should already be properly typed from download
    ds = xr.open_zarr(path, consolidated=True)

    # Validate time coordinate types
    if "init_time" in ds.coords:
        if ds.coords["init_time"].dtype != np.dtype("datetime64[ns]"):
            logging.warning(
                f"init_time has unexpected dtype {ds.coords['init_time'].dtype}, expected datetime64[ns]"
            )

    if "lead_time" in ds.coords:
        if not np.issubdtype(ds.coords["lead_time"].dtype, np.timedelta64):
            logging.warning(
                f"lead_time has unexpected dtype {ds.coords['lead_time'].dtype}, expected timedelta64[ns]"
            )

    return ds


def combine_and_write(items: List[Tuple[datetime, str]], out_path: str, label: str) -> bool:
    if not items:
        logging.info(f"No {label} archives found to combine.")
        return False
    logging.info(f"Combining {len(items)} {label} archives into {out_path}")
    dsets: List[xr.Dataset] = []
    for dt, p in items:
        try:
            ds = open_and_tag(p, dt)
            dsets.append(ds)
            logging.info(f"  included {label}: {p}")
        except Exception as e:
            logging.error(f"Failed to open {p}: {e}")
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

        # Rechunk to ensure proper alignment for Zarr writing
        # Determine if this is ensemble data (has 'ensemble' dimension) or control data
        has_ensemble = "ensemble" in combined.dims

        # Build chunking dict based on available dimensions
        chunk_dict = {
            "init_time": 1,
            "lead_time": 1,
            "latitude": -1,
            "longitude": -1,
        }

        if has_ensemble:
            # Chunk ensemble dimension as full size (not by 10)
            chunk_dict["ensemble"] = -1

        # Only add 'level' if it exists in dimensions
        if "level" in combined.dims:
            chunk_dict["level"] = 1

        # Rechunk the dataset
        logging.info(f"Rechunking with: {chunk_dict}")
        combined = combined.chunk(chunk_dict)

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
            # Clean up to avoid schema conflicts
            import shutil

            shutil.rmtree(out_path)
        combined.to_zarr(out_path, mode="w", consolidated=True, zarr_format=2)
        logging.info(f"Wrote combined archive: {out_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to write combined {label} archive: {e}")
        import traceback

        traceback.print_exc()
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
