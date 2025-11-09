#!/usr/bin/env python3
"""
Download IFS data (ESFM only) for a single date with improved error handling and logging.
This script can be used to download individual dates or as part of a larger bulk process.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Mapping

import earthkit.data
import numpy as np
import xarray as xr
from earthkit.data import settings

# Model name can be overridden via environment variable MODEL_NAME (default: esfm)
MODEL_NAME = os.getenv("MODEL_NAME", "esfm")
ESFM_FIELDS = {
    "grid": [0.25, 0.25],
    "area": [90, -180, -90, 180],  # Global
    # Pressure levels and parameters
    "pressure_levels": [
        1000,
        925,
        850,
        700,
        600,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        50,
    ],
    "pressure_level_params": ["u", "v", "t", "q", "z"],
    # Single level parameters
    "single_level_params": ["2t", "10u", "10v", "msl", "tp", "z"],
}


def get_debug_fields():
    """Return a greatly reduced set of fields for fast debug runs."""
    return {
        "grid": [1.0, 1.0],  # coarser grid
        "area": [60, -30, 30, 30],  # small region (N/W/S/E)
        "pressure_levels": [1000, 925],  # single pressure level
        "pressure_level_params": ["u"],  # single PL variable
        "single_level_params": ["2t"],  # single SL variable
    }


def setup_logging(log_file=None):
    """Setup logging configuration."""
    level = logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    if log_file:
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=level, format=format_str)


def create_fields_file(output_path, fields):
    """Create a fields.txt file with the parameters used for this run."""

    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    fields_content = f"""grid: {fields["grid"]}
area: {fields["area"]}

pressure_levels: {fields["pressure_levels"]}
pressure_level_params: {fields["pressure_level_params"]}

single_level_params: {fields["single_level_params"]}
"""

    fields_path = os.path.join(output_path, "fields.txt")
    with open(fields_path, "w") as f:
        f.write(fields_content)

    logging.info(f"Created fields.txt at {fields_path}")
    return fields_path


def _to_serializable(value: Any) -> Any:
    """Best-effort conversion of attribute values into Zarr-safe JSON types.

    Zarr requires attributes to be JSON-serializable. This function converts
    common non-serializable types (numpy scalars, datetimes, numpy dtypes, etc.)
    to safe representations. Anything unknown falls back to str(value).
    """
    # Basic safe types
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # numpy scalars -> Python scalars
    if isinstance(value, (np.generic,)):
        try:
            return value.item()
        except Exception:
            return str(value)
    # numpy dtype
    if isinstance(value, (np.dtype,)):
        return str(value)
    # datetime / datetime64 -> ISO string
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (np.datetime64,)):
        try:
            return np.datetime_as_string(value, timezone="naive")
        except Exception:
            return str(value)
    # lists/tuples
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    # dict-like
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    # Fallback
    return str(value)


def sanitize_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Sanitize Dataset and Variable attrs to be Zarr/JSON-safe.

    This mutates a shallow copy of the Dataset's attrs and variables' attrs to
    ensure they are JSON-serializable when writing to Zarr.
    """
    # Sanitize dataset attrs
    if ds.attrs:
        ds.attrs = {str(k): _to_serializable(v) for k, v in ds.attrs.items()}
    # Sanitize variable and coord attrs
    for var_name in list(ds.data_vars) + list(ds.coords):
        var = ds[var_name]
        if var.attrs:
            var.attrs = {str(k): _to_serializable(v) for k, v in var.attrs.items()}
    return ds


def log_ds_summary(name: str, ds: xr.Dataset):
    """Log a concise summary of a Dataset without dumping data."""
    try:
        sizes = {k: int(v) for k, v in ds.sizes.items()}
        coords = list(ds.coords)
        data_vars = list(ds.data_vars)
        chunks = getattr(ds, "chunks", None)
        attr_keys = list(ds.attrs.keys()) if ds.attrs else []
        logging.info(f"{name} summary: dims={sizes}, coords={coords}, data_vars={data_vars}")
        logging.info(f"{name} chunks: {chunks}")
        logging.info(f"{name} attr keys: {attr_keys}")
    except Exception as e:
        logging.warning(f"Failed to log dataset summary for {name}: {e}")


def rename_and_enrich(ds: xr.Dataset, *, has_ensemble: bool, init_dt: datetime) -> xr.Dataset:
    """Rename dims/coords to desired schema and add init_time dimension/coord.

    - number -> ensemble (if present and has_ensemble=True)
    - step -> lead_time (if present)
    - init_time: always added from init_dt; drop any existing 'time' coord
    - Rename variables to full descriptive names
    """
    rename_map = {}
    if has_ensemble and ("number" in ds.dims or "number" in ds.coords):
        rename_map["number"] = "ensemble"
    if "step" in ds.dims or "step" in ds.coords:
        rename_map["step"] = "lead_time"
    if rename_map:
        ds = ds.rename(rename_map)

    # Rename variables to full descriptive names
    var_rename_map = {
        "10u": "10m_u_component_of_wind",
        "10v": "10m_v_component_of_wind",
        "2t": "2m_temperature",
        "msl": "mean_sea_level_pressure",
        "tp": "total_precipitation",
        "z": "geopotential",
        "q": "specific_humidity",
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
    }
    var_rename_dict = {old: new for old, new in var_rename_map.items() if old in ds.data_vars}
    if var_rename_dict:
        ds = ds.rename(var_rename_dict)

    # Normalize init_time: drop any existing 'time' coord and set our own anchor
    ts = np.datetime64(init_dt, "ns")
    ds = ds.drop_vars(["time"], errors="ignore")
    if "init_time" not in ds.dims:
        ds = ds.expand_dims(init_time=[ts])
    else:
        # Ensure it is a dimension and length-1
        if ds.sizes.get("init_time", 0) != 1:
            ds = ds.isel(init_time=0)
            ds = ds.expand_dims(init_time=[ts])
    # Ensure coord value is exactly the provided ts
    ds = ds.assign_coords(init_time=[ts])

    return ds


def cast_float32(ds: xr.Dataset) -> xr.Dataset:
    """Cast all floating data_vars and latitude/longitude coords to float32."""
    # Data variables
    for v in list(ds.data_vars):
        da = ds[v]
        if np.issubdtype(da.dtype, np.floating) and da.dtype != np.float32:
            ds[v] = da.astype(np.float32)
    # Coordinates: latitude/longitude
    for cname in ("latitude", "longitude"):
        if cname in ds.coords:
            ca = ds.coords[cname]
            if np.issubdtype(ca.dtype, np.floating) and ca.dtype != np.float32:
                ds = ds.assign_coords({cname: ca.astype(np.float32)})
    return ds


def normalize_longitudes(ds: xr.Dataset) -> xr.Dataset:
    """Ensure longitudes are in the 0..360 range and sorted ascending.

    If longitudes are in -180..180, converts via (lon % 360) then sorts by longitude,
    reordering data along that dimension lazily.
    """
    if "longitude" not in ds.coords:
        return ds
    lon = ds["longitude"]
    try:
        lon_min = float(lon.min())
        lon_max = float(lon.max())
    except Exception:
        return ds
    # Detect typical -180..180 range
    if lon_min < 0.0 or lon_max <= 180.0:
        lon360 = (
            (lon % 360).astype(np.float32) if np.issubdtype(lon.dtype, np.floating) else (lon % 360)
        )
        ds = ds.assign_coords(longitude=lon360)
        ds = ds.sortby("longitude")
    return ds


def normalize_time_encodings(ds: xr.Dataset) -> xr.Dataset:
    """Normalize time coordinates for robust Zarr round-trips.

    - init_time: enforce dtype datetime64[ns] and clear CF hints (units/calendar).
    - lead_time: enforce dtype timedelta64[ns]. If it's numeric, interpret as hours.
    """
    ds = ds.copy()

    # init_time: ensure datetime64[ns] and clear CF metadata that might trigger CF-decoding
    if "init_time" in ds.coords:
        try:
            ds["init_time"] = ds["init_time"].astype("datetime64[ns]")
            # Set dtype explicitly in encoding to prevent xarray from converting to int64
            ds["init_time"].encoding = {"dtype": "datetime64[ns]"}
            for k in ("units", "calendar"):
                if k in ds["init_time"].attrs:
                    ds["init_time"].attrs.pop(k, None)
        except Exception:
            pass

    # lead_time: ensure timedelta64[ns]
    if "lead_time" in ds.coords:
        lt = ds["lead_time"]
        try:
            if np.issubdtype(lt.dtype, np.timedelta64):
                # Normalize to ns resolution
                ds["lead_time"] = lt.astype("timedelta64[ns]")
            else:
                # Treat numeric lead_time as hours by convention (IFS step is in hours)
                # Works for both integer and float inputs.
                lt_int = lt.astype("int64")
                td = (lt_int * np.timedelta64(1, "h")).astype("timedelta64[ns]")
                ds = ds.assign_coords(lead_time=td)
            # Clear any encoding that might prompt CF conversions back to integers
            # Set dtype explicitly in encoding to prevent xarray from converting to int64
            ds["lead_time"].encoding = {"dtype": "timedelta64[ns]"}
            # Also drop confusing attrs like units so xarray doesn't attempt CF decode
            for k in ("units", "calendar"):
                if k in ds["lead_time"].attrs:
                    ds["lead_time"].attrs.pop(k, None)
        except Exception:
            pass

    return ds


def download_ifs_ensemble(
    output_dir, date_time, num_days, interval=6, *, fields, debug_small=False
):
    """Download IFS ensemble data for a specific date."""

    # Set cache directory
    cache_dir = os.path.join(output_dir, ".earthkit-cache")
    os.makedirs(cache_dir, exist_ok=True)
    settings.set("user-cache-directory", cache_dir)

    # Setup paths
    date_str = date_time.strftime("%Y%m%d%H%M")
    path = os.path.join(output_dir, date_str, MODEL_NAME)
    output_file = os.path.join(path, "ifs_ens.zarr")

    # NOTE: Do not early-return on existence; we support resuming by appending,
    # but avoid unnecessary downloads/conversions when the store is already complete.
    output_exists = os.path.exists(output_file)

    # Create fields.txt if it doesn't exist
    if not os.path.exists(os.path.join(path, "fields.txt")):
        create_fields_file(path, fields)

    try:
        # Use the provided fields for this run
        grid = fields["grid"]
        area = fields["area"]
        pressure_levels = fields["pressure_levels"]
        pressure_level_params = fields["pressure_level_params"]
        single_level_params = fields["single_level_params"]

        # Extract date components
        year = date_time.strftime("%Y")
        month = date_time.strftime("%m")
        day = date_time.strftime("%d")
        hour = date_time.strftime("%H")

        logging.info(f"Downloading ensemble data for {date_str} ({num_days} days)")

        # Define chunking separately for pressure-level and single-level data.
        # Important: Avoid using keys that may not exist as dimensions (e.g., 'time', 'surface').
        # We hardcode the pressure-level dimension name to 'level'.
        # Do not chunk along ensemble/number (use -1 => single chunk for full dim)
        chunks_pl = {
            "number": -1,
            "step": 1,
            "level": 1,
            "latitude": -1,
            "longitude": -1,
        }
        chunks_surface = {
            "number": -1,
            "step": 1,
            "latitude": -1,
            "longitude": -1,
        }

        # Expected ensemble member count
        expected_ensemble = 2 if debug_small else 50

        # Early completeness check: if Zarr exists with full ensemble and surface '2t', skip entirely
        if output_exists:
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                existing_count = int(existing.sizes.get("ensemble", 0))
                has_surface_t2 = "2t" in existing.data_vars
                if existing_count >= expected_ensemble and has_surface_t2:
                    logging.info(
                        "Existing ensemble store appears complete (ensemble=%d, has 2t). Skipping.",
                        existing_count,
                    )
                    return True
            except Exception as e:
                logging.warning(
                    f"Failed to open existing store for completeness check: {e}. Will proceed."
                )

        # Split the "number" dimension into chunks for pressure-level data
        number_chunks = (
            ["1/to/2/by/1"]
            if debug_small
            else [f"{i}/to/{min(i + 9, 50)}/by/1" for i in range(1, 51, 10)]
        )

        # Determine resume point by inspecting existing ensemble members in Zarr
        start_chunk_index = 0
        if output_exists:
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                existing_count = int(existing.sizes.get("ensemble", 0))

                # Compute per-chunk sizes and find exact prefix match
                chunk_sizes = []
                for spec in number_chunks:
                    # spec like "a/to/b/by/1"
                    try:
                        a = int(spec.split("/to/")[0])
                        rest = spec.split("/to/")[1]
                        b = int(rest.split("/by/")[0])
                        s = int(rest.split("/by/")[1])
                        size = ((b - a) // s) + 1
                    except Exception:
                        size = 0
                    chunk_sizes.append(size)
                cum = 0
                start_chunk_index = 0
                exact = False
                for idx, cs in enumerate(chunk_sizes):
                    if cum + cs == existing_count:
                        start_chunk_index = idx + 1
                        exact = True
                        break
                    elif cum + cs > existing_count:
                        # Non-aligned partial write detected
                        exact = False
                        break
                    cum += cs

                if existing_count == 0:
                    start_chunk_index = 0
                    exact = True

                if not exact and existing_count > 0:
                    logging.warning(
                        "Existing ensemble store is not aligned to chunk boundaries (found %d members). "
                        "Restarting this date from scratch.",
                        existing_count,
                    )
                    # Remove existing store to avoid duplication and restart
                    import shutil

                    shutil.rmtree(output_file, ignore_errors=True)
                    output_exists = False
                    start_chunk_index = 0
            except Exception as e:
                logging.warning(f"Failed to inspect existing ensemble store for resume: {e}")
                start_chunk_index = 0

        # Retrieve the pressure level data in chunks
        for i, number_chunk in enumerate(number_chunks):
            if i < start_chunk_index:
                logging.info(
                    f"Skipping already completed ensemble chunk {i + 1}/{len(number_chunks)}"
                )
                continue
            logging.info("Downloading pressure level data...")
            logging.info(
                f"Processing ensemble chunk {i + 1}/{len(number_chunks)}: members {number_chunk}"
            )

            # Build the request for pressure level data for this chunk
            request_pl = {
                "area": area,
                "class": "od",
                "date": f"{year}-{month}-{day}",
                "expver": "1",
                "grid": grid,
                "levtype": "pl",
                "levelist": pressure_levels,
                "param": pressure_level_params,
                "number": number_chunk,
                "step": "0/1" if debug_small else f"0/to/{num_days * 24}/by/{interval}",
                "stream": "enfo",
                "expect": "any",
                "time": hour,
                "type": "pf",
            }

            ds_pressure_chunk = earthkit.data.from_source("mars", request_pl, lazily=True)

            shortnames = list(set(ds_pressure_chunk.metadata("shortName")))
            # Some requests may not include relative humidity ('r'); handle gracefully
            has_r = "r" in shortnames
            normal_vars = [var for var in shortnames if var != "r"]

            logging.info("Converting pressure level data to xarray...")
            # Convert to xarray and chunk with hardcoded 'level' dimension
            ds_normal = ds_pressure_chunk.sel(shortName=normal_vars).to_xarray(chunks=chunks_pl)
            if has_r:
                ds_special = ds_pressure_chunk.sel(shortName=["r"]).to_xarray(chunks=chunks_pl)
                ds_combined = xr.merge([ds_normal, ds_special])
            else:
                ds_combined = ds_normal

            ds_combined = ds_combined.chunk(chunks_pl).drop_vars("valid_time", errors="ignore")
            # Rename/cast/time-encoding per-range
            ds_combined = rename_and_enrich(ds_combined, has_ensemble=True, init_dt=date_time)
            ds_combined = cast_float32(ds_combined)
            ds_combined = normalize_longitudes(ds_combined)
            ds_combined = normalize_time_encodings(ds_combined)
            if debug_small:
                log_ds_summary(f"ensemble.pl.chunk{i + 1}.renamed", ds_combined)

            logging.info("Writing pressure level data to zarr...")
            # Write to Zarr
            # Sanitize attributes prior to writing
            ds_to_write = sanitize_dataset_attrs(ds_combined)
            # Choose write mode depending on resume point
            write_mode = "a" if (output_exists or i > 0 or start_chunk_index > 0) else "w"
            ds_to_write.to_zarr(
                output_file,
                consolidated=True,
                mode=write_mode,
                zarr_format=2,
                append_dim="ensemble"
                if (output_exists or i > 0 or start_chunk_index > 0)
                else None,
            )

        # Add surface data (idempotent): only download/convert if needed
        write_surface = True
        if os.path.exists(output_file):
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                # '2t' is a surface-only variable we write; if present, surface already added
                if "2t" in existing.data_vars:
                    write_surface = False
            except Exception:
                # If inspection fails, fall back to writing (safe overwrite with mode="a")
                write_surface = True

        if write_surface:
            logging.info("Downloading surface data...")
            # Build the request for single level data (all ensemble members)
            request_sfc = {
                "area": area,
                "class": "od",
                "date": f"{year}-{month}-{day}",
                "expver": "1",
                "grid": grid,
                "levtype": "sfc",
                "number": "1/to/2/by/1" if debug_small else "1/to/50/by/1",
                "param": single_level_params,
                "step": "0/1" if debug_small else f"0/to/{num_days * 24}/by/{interval}",
                "stream": "enfo",
                "expect": "any",
                "time": hour,
                "type": "pf",
            }

            ds_single = earthkit.data.from_source("mars", request_sfc, lazily=True)
            logging.info("Converting surface data to xarray...")
            ds_single = (
                ds_single.to_xarray(chunks=chunks_surface)
                .drop_vars("valid_time", errors="ignore")
                .chunk(chunks_surface)
            )
            # Rename/cast per-range
            ds_single = rename_and_enrich(ds_single, has_ensemble=True, init_dt=date_time)
            ds_single = cast_float32(ds_single)
            ds_single = normalize_longitudes(ds_single)
            if debug_small:
                log_ds_summary("ensemble.surface.renamed", ds_single)

            logging.info("Adding surface data to zarr...")
            ds_single_sanitized = sanitize_dataset_attrs(
                ds_single.drop_vars(["z"], errors="ignore")
            )
            ds_single_sanitized = normalize_time_encodings(ds_single_sanitized)
            ds_single_sanitized = normalize_longitudes(ds_single_sanitized)
            ds_single_sanitized.to_zarr(output_file, consolidated=True, zarr_format=2, mode="a")
        else:
            logging.info("Surface data already present; skipping surface write.")

        logging.info(f"Successfully downloaded ensemble data to {output_file}")
        return True

    except Exception as e:
        logging.error(f"Failed to download ensemble data for {date_str}: {e}")
        return False


def download_ifs_control(output_dir, date_time, num_days, interval=6, *, fields, debug_small=False):
    """Download IFS control data for a specific date.

    Note: For consistency with the ensemble workflow, we always process
    pressure-level data first and then append single-level (surface) data.
    """

    # Set cache directory
    cache_dir = os.path.join(output_dir, ".earthkit-cache")
    os.makedirs(cache_dir, exist_ok=True)
    settings.set("user-cache-directory", cache_dir)

    # Setup paths
    date_str = date_time.strftime("%Y%m%d%H%M")
    path = os.path.join(output_dir, date_str, MODEL_NAME)
    output_file = os.path.join(path, "ifs_control.zarr")

    # Check if already exists
    if os.path.exists(output_file):
        logging.info(f"Control data already exists: {output_file}")
        return True

    # Create fields.txt if it doesn't exist
    if not os.path.exists(os.path.join(path, "fields.txt")):
        create_fields_file(path, fields)

    try:
        # Use the provided fields for this run
        grid = fields["grid"]
        area = fields["area"]
        pressure_levels = fields["pressure_levels"]
        pressure_level_params = fields["pressure_level_params"]
        single_level_params = fields["single_level_params"]

        # Extract date components
        year = date_time.strftime("%Y")
        month = date_time.strftime("%m")
        day = date_time.strftime("%d")
        hour = date_time.strftime("%H")

        logging.info(f"Downloading control data for {date_str} ({num_days} days)")

        # Define chunking for control run; avoid non-existent dims like 'time' and 'surface'.
        # We hardcode the pressure-level dimension name to 'level'.
        chunks_pl = {
            "step": 1,
            "level": 1,
            "latitude": -1,
            "longitude": -1,
        }
        chunks_surface = {
            "step": 1,
            "latitude": -1,
            "longitude": -1,
        }
        # 1) Retrieve and write the pressure-level data first
        logging.info("Downloading pressure level data...")
        request_pl = {
            "area": area,
            "class": "od",
            "date": f"{year}-{month}-{day}",
            "expver": "1",
            "grid": grid,
            "levtype": "pl",
            "levelist": pressure_levels,
            "param": pressure_level_params,
            "step": "0/1" if debug_small else f"0/to/{num_days * 24}/by/{interval}",
            "stream": "enfo",
            "expect": "any",
            "time": hour,
            "type": "cf",
        }

        ds_pressure = earthkit.data.from_source("mars", request_pl, lazily=True)

        shortnames = list(set(ds_pressure.metadata("shortName")))
        has_r = "r" in shortnames
        normal_vars = [var for var in shortnames if var != "r"]

        ds_normal = ds_pressure.sel(shortName=normal_vars).to_xarray(chunks=chunks_pl)
        if has_r:
            ds_special = ds_pressure.sel(shortName=["r"]).to_xarray(chunks=chunks_pl)
            ds_combined = xr.merge([ds_normal, ds_special])
        else:
            ds_combined = ds_normal

        ds_combined = ds_combined.chunk(chunks_pl).drop_vars("valid_time", errors="ignore")
        ds_combined = rename_and_enrich(ds_combined, has_ensemble=False, init_dt=date_time)
        ds_combined = cast_float32(ds_combined)
        ds_combined = normalize_longitudes(ds_combined)
        ds_combined = normalize_time_encodings(ds_combined)
        if debug_small:
            log_ds_summary("control.pl.renamed", ds_combined)

        # Write pressure level data to zarr
        logging.info("Writing pressure level data to zarr...")
        ds_to_write = sanitize_dataset_attrs(ds_combined)
        ds_to_write.to_zarr(output_file, consolidated=True, zarr_format=2, mode="w")

        # 2) Retrieve and append the single-level (surface) data
        logging.info("Downloading surface data...")
        request_sfc = {
            "area": area,
            "class": "od",
            "date": f"{year}-{month}-{day}",
            "expver": "1",
            "grid": grid,
            "levtype": "sfc",
            "param": single_level_params,
            "step": "0/1" if debug_small else f"0/to/{num_days * 24}/by/{interval}",
            "stream": "enfo",
            "expect": "any",
            "time": hour,
            "type": "cf",
        }
        ds_single = earthkit.data.from_source("mars", request_sfc, lazily=True)
        ds_single = (
            ds_single.to_xarray(chunks=chunks_surface)
            .drop_vars("valid_time", errors="ignore")
            .chunk(chunks_surface)
        )
        # Rename/cast per-range (no ensemble dim expected for control)
        ds_single = rename_and_enrich(ds_single, has_ensemble=False, init_dt=date_time)
        ds_single = cast_float32(ds_single)
        ds_single = normalize_longitudes(ds_single)
        if debug_small:
            log_ds_summary("control.surface.renamed", ds_single)

        logging.info("Adding surface data...")
        ds_single_sanitized = sanitize_dataset_attrs(ds_single.drop_vars(["z"], errors="ignore"))
        ds_single_sanitized = normalize_time_encodings(ds_single_sanitized)
        ds_single_sanitized = normalize_longitudes(ds_single_sanitized)
        ds_single_sanitized.to_zarr(output_file, consolidated=True, zarr_format=2, mode="a")

        logging.info(f"Successfully downloaded control data to {output_file}")
        return True

    except Exception as e:
        logging.error(f"Failed to download control data for {date_str}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download IFS data for a single date")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("date_time", type=str, help="Date and time in format YYYYMMDDHHMM")
    parser.add_argument("num_days", type=int, help="Number of days to download")
    # Model name is fixed to ESFM; no CLI option needed
    parser.add_argument("--interval", type=int, default=6, help="Time step in hours")
    parser.add_argument(
        "--debug-small",
        action="store_true",
        help="Download a tiny subset (small area/grid, few variables/levels, minimal steps)",
    )
    parser.add_argument(
        "--download-type",
        type=str,
        choices=["ensemble", "control", "both"],
        default="both",
        help="Type of data to download",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    # Parse date
    try:
        date_time = datetime.strptime(args.date_time, "%Y%m%d%H%M")
    except ValueError:
        logging.error(f"Invalid date format: {args.date_time}. Use YYYYMMDDHHMM")
        return 1

    logging.info(f"Starting download for {args.date_time}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {MODEL_NAME}")
    logging.info(f"Number of days: {args.num_days}")
    logging.info(f"Download type: {args.download_type}")
    logging.info(f"Debug small: {args.debug_small}")

    # Choose fields for this run
    fields = get_debug_fields() if args.debug_small else ESFM_FIELDS

    success = True

    # Download ensemble data
    if args.download_type in ["ensemble", "both"]:
        success &= download_ifs_ensemble(
            args.output_dir,
            date_time,
            args.num_days,
            args.interval,
            fields=fields,
            debug_small=args.debug_small,
        )

    # Download control data
    if args.download_type in ["control", "both"]:
        success &= download_ifs_control(
            args.output_dir,
            date_time,
            args.num_days,
            args.interval,
            fields=fields,
            debug_small=args.debug_small,
        )

    if success:
        logging.info(f"Successfully completed download for {args.date_time}")
        return 0
    else:
        logging.error(f"Download failed for {args.date_time}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
