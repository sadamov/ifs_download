#!/usr/bin/env python3
"""
Download IFS data (ESFM only) for a single date with improved error handling and logging.
This script can be used to download individual dates or as part of a larger bulk process.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

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


def build_number_chunks(total_members: int, chunk_size: int):
    """Return inclusive member ranges capped by chunk_size."""
    chunk_size = max(1, min(chunk_size, total_members))
    ranges = []
    start = 1
    while start <= total_members:
        end = min(start + chunk_size - 1, total_members)
        ranges.append((start, end))
        start = end + 1
    return ranges


def generate_init_times(
    start_dt: datetime, end_dt: datetime | None, interval_hours: int
) -> list[datetime]:
    """Generate inclusive init_time list stepping by interval_hours."""
    if end_dt is None:
        return [start_dt]
    if end_dt < start_dt:
        raise ValueError("range_end must be >= range_start")
    step = timedelta(hours=max(1, interval_hours))
    inits: list[datetime] = []
    current = start_dt
    while current <= end_dt:
        inits.append(current)
        current += step
    return inits


def group_init_times_by_date(init_times: Sequence[datetime]) -> list[dict[str, Any]]:
    """Group init_times by calendar date while preserving chronological order."""
    groups: dict[str, list[datetime]] = {}
    for dt in init_times:
        key = dt.strftime("%Y-%m-%d")
        groups.setdefault(key, []).append(dt)
    ordered: list[dict[str, Any]] = []
    for date_str, values in groups.items():
        values.sort()
        hours = sorted({dt.strftime("%H") for dt in values})
        ordered.append({"date": date_str, "datetimes": values[:], "hours": hours})
    return ordered


def to_datetime64(values: Iterable[datetime]) -> np.ndarray:
    """Convert iterable of datetimes to numpy datetime64[ns] array."""
    return np.array([np.datetime64(dt, "ns") for dt in values], dtype="datetime64[ns]")


def dedupe_sorted_datetimes(values: Sequence[datetime]) -> list[datetime]:
    """Return sorted datetimes with duplicates removed while preserving order."""
    ordered = sorted(values)
    unique: list[datetime] = []
    for dt in ordered:
        if not unique or dt != unique[-1]:
            unique.append(dt)
    return unique


def order_init_time(ds: xr.Dataset, target_values: np.ndarray) -> xr.Dataset:
    """Reorder/select init_time coordinate to match the provided values."""
    if "init_time" not in ds.coords:
        raise ValueError("Dataset missing init_time coordinate after conversion")
    available = ds["init_time"].values.astype("datetime64[ns]")
    indices: list[int] = []
    missing: list[np.datetime64] = []
    for val in target_values:
        matches = np.where(available == val)[0]
        if matches.size == 0:
            missing.append(val)
        else:
            indices.append(int(matches[0]))
    if missing:
        raise ValueError(f"Missing init_time values in dataset: {missing}")
    ds = ds.isel(init_time=indices)
    ds = ds.assign_coords(init_time=target_values)
    return ds


def init_coord_matches(coord: xr.DataArray, init_times: Sequence[datetime]) -> bool:
    """Check whether coord covers the provided init_times (ignoring order)."""
    if coord.size == 0:
        return False
    existing = np.sort(coord.astype("datetime64[ns]").values)
    target = np.sort(to_datetime64(init_times))
    return existing.shape == target.shape and np.array_equal(existing, target)


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


def rename_and_enrich(
    ds: xr.Dataset,
    *,
    has_ensemble: bool,
    init_times: Sequence[datetime] | None = None,
    add_dummy_ensemble: bool = False,
) -> xr.Dataset:
    """Rename dims/coords to desired schema and add init_time dimension/coord.

    - number -> ensemble (if present and has_ensemble=True)
    - step -> lead_time (if present)
    - init_time: always added from init_dt; drop any existing 'time' coord
    - Rename variables to full descriptive names
    - Optionally add an ensemble dimension of size 1 for control data
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

    if "time" in ds.dims or "time" in ds.coords:
        ds = ds.rename({"time": "init_time"})
    ds = ds.drop_vars(["time"], errors="ignore")

    if "init_time" not in ds.dims:
        if not init_times:
            raise ValueError("init_times must be provided when dataset lacks init_time coord")
        ts = to_datetime64(init_times)
        ds = ds.expand_dims(init_time=ts)
        ds = ds.assign_coords(init_time=ts)
    else:
        ds["init_time"] = ds["init_time"].astype("datetime64[ns]")
        if init_times:
            target = to_datetime64(init_times)
            ds = order_init_time(ds, target)
        else:
            ds = ds.assign_coords(init_time=ds["init_time"].astype("datetime64[ns]"))

    if add_dummy_ensemble and "ensemble" not in ds.dims:
        ds = ds.expand_dims(ensemble=[0])
        # Keep ensemble leading for consistency with true ensembles
        ordered_dims = ["ensemble"] + [dim for dim in ds.dims if dim != "ensemble"]
        ds = ds.transpose(*ordered_dims)

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


def normalize_latitudes(ds: xr.Dataset, *, descending: bool = True) -> xr.Dataset:
    """Sort latitude coordinate to a consistent orientation (default: 90 -> -90)."""
    if "latitude" not in ds.coords:
        return ds
    lat = ds["latitude"]
    try:
        values = lat.values
    except Exception:
        return ds
    if values.size <= 1:
        return ds
    diffs = np.diff(values)
    if descending:
        if np.all(diffs <= 0):
            return ds
        return ds.sortby("latitude", ascending=False)
    if np.all(diffs >= 0):
        return ds
    return ds.sortby("latitude", ascending=True)


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
    output_dir, init_times, num_days, interval=6, *, fields, debug_small=False
):
    """Download IFS ensemble data for one or more init_times."""

    if not init_times:
        logging.warning("No init_times requested for ensemble download; skipping.")
        return True

    ordered_inits = dedupe_sorted_datetimes(init_times)
    date_str = ordered_inits[0].strftime("%Y%m%d%H%M")
    init_summary = (
        f"{ordered_inits[0]:%Y-%m-%d %H:%M} -> {ordered_inits[-1]:%Y-%m-%d %H:%M}"
        if len(ordered_inits) > 1
        else ordered_inits[0].strftime("%Y-%m-%d %H:%M")
    )
    expected_init_count = len(ordered_inits)
    date_groups = group_init_times_by_date(ordered_inits)
    if not date_groups:
        logging.warning("Unable to group init_times for ensemble download; nothing to do.")
        return True

    # Set cache directory
    cache_dir = os.path.join(output_dir, ".earthkit-cache")
    os.makedirs(cache_dir, exist_ok=True)
    settings.set("user-cache-directory", cache_dir)

    # Setup paths
    path = os.path.join(output_dir, date_str, MODEL_NAME)
    output_file = os.path.join(path, "ifs_ens.zarr")

    # NOTE: Do not early-return on existence; we support resuming by appending,
    # but avoid unnecessary downloads/conversions when the store is already complete.
    output_exists = os.path.exists(output_file)

    # Create fields.txt if it doesn't exist
    if not os.path.exists(os.path.join(path, "fields.txt")):
        create_fields_file(path, fields)

    try:
        grid = fields["grid"]
        area = fields["area"]
        pressure_levels = fields["pressure_levels"]
        pressure_level_params = fields["pressure_level_params"]
        single_level_params = fields["single_level_params"]

        max_lead_hours = int(os.getenv("MAX_LEAD_TIME_HOURS", str(num_days * 24)))
        if max_lead_hours % interval != 0:
            max_lead_hours = (max_lead_hours // interval) * interval
        logging.info(
            "Downloading ensemble data for %s (%d init_times, interval %dh, max lead %dh)",
            init_summary,
            expected_init_count,
            interval,
            max_lead_hours,
        )

        # Define chunking separately for pressure-level and single-level data.
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

        expected_ensemble = 2 if debug_small else 50

        # Early completeness check: require both ensemble count and init_time coverage
        if output_exists:
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                existing_count = int(existing.sizes.get("ensemble", 0))
                has_surface_t2 = "2t" in existing.data_vars
                has_all_init = "init_time" in existing.coords and init_coord_matches(
                    existing["init_time"], ordered_inits
                )
                existing.close()
                if existing_count >= expected_ensemble and has_surface_t2 and has_all_init:
                    logging.info(
                        "Existing ensemble store appears complete (ensemble=%d, init_times=%d). Skipping.",
                        existing_count,
                        expected_init_count,
                    )
                    return True
            except Exception as e:
                logging.warning(
                    "Failed to open existing ensemble store for completeness check: %s. Will proceed.",
                    e,
                )

        default_chunk_size = expected_ensemble
        chunk_size_env = os.getenv("ENSEMBLE_MARS_CHUNK_SIZE")
        if chunk_size_env:
            try:
                default_chunk_size = int(chunk_size_env)
            except ValueError:
                logging.warning(
                    "Invalid ENSEMBLE_MARS_CHUNK_SIZE=%s, falling back to %d",
                    chunk_size_env,
                    expected_ensemble,
                )
                default_chunk_size = expected_ensemble
        if default_chunk_size <= 0:
            logging.warning(
                "ENSEMBLE_MARS_CHUNK_SIZE must be positive; received %d. Using %d instead.",
                default_chunk_size,
                expected_ensemble,
            )
            default_chunk_size = expected_ensemble

        number_chunks = build_number_chunks(expected_ensemble, default_chunk_size)
        used_chunk_size = number_chunks[0][1] - number_chunks[0][0] + 1 if number_chunks else 0
        logging.info(
            "Ensemble chunking: %d member(s) per MARS request -> %d request(s) total",
            used_chunk_size,
            len(number_chunks),
        )

        start_chunk_index = 0
        if output_exists:
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                existing_count = int(existing.sizes.get("ensemble", 0))
                chunk_sizes = [(end - start + 1) for start, end in number_chunks]
                cum = 0
                exact = False
                for idx, cs in enumerate(chunk_sizes):
                    if cum + cs == existing_count:
                        start_chunk_index = idx + 1
                        exact = True
                        break
                    elif cum + cs > existing_count:
                        exact = False
                        break
                    cum += cs
                if existing_count == 0:
                    start_chunk_index = 0
                    exact = True
                if not exact and existing_count > 0:
                    logging.warning(
                        "Existing ensemble store is not aligned to chunk boundaries (found %d members). "
                        "Restarting this range from scratch.",
                        existing_count,
                    )
                    import shutil

                    shutil.rmtree(output_file, ignore_errors=True)
                    output_exists = False
                    start_chunk_index = 0
                existing.close()
            except Exception as e:
                logging.warning(f"Failed to inspect existing ensemble store for resume: {e}")
                start_chunk_index = 0

        # Retrieve the pressure level data in chunks across all requested dates
        for i, (chunk_start, chunk_end) in enumerate(number_chunks):
            if i < start_chunk_index:
                logging.info(
                    f"Skipping already completed ensemble chunk {i + 1}/{len(number_chunks)}"
                )
                continue
            logging.info(
                "Processing ensemble chunk %d/%d: members %d-%d across %d day(s)",
                i + 1,
                len(number_chunks),
                chunk_start,
                chunk_end,
                len(date_groups),
            )

            chunk_dsets: list[xr.Dataset] = []
            for group in date_groups:
                hours_token = "/".join(group["hours"])
                request_pl = {
                    "area": area,
                    "class": "od",
                    "date": group["date"],
                    "expver": "1",
                    "grid": grid,
                    "levtype": "pl",
                    "levelist": pressure_levels,
                    "param": pressure_level_params,
                    "number": f"{chunk_start}/to/{chunk_end}/by/1",
                    "step": "0/1"
                    if debug_small
                    else f"{interval}/to/{max_lead_hours}/by/{interval}",
                    "stream": "enfo",
                    "expect": "any",
                    "time": hours_token,
                    "type": "pf",
                }

                ds_pressure_chunk = earthkit.data.from_source("mars", request_pl, lazily=True)
                shortnames = list(set(ds_pressure_chunk.metadata("shortName")))
                has_r = "r" in shortnames
                normal_vars = [var for var in shortnames if var != "r"]

                ds_normal = ds_pressure_chunk.sel(shortName=normal_vars).to_xarray(chunks=chunks_pl)
                if has_r:
                    ds_special = ds_pressure_chunk.sel(shortName=["r"]).to_xarray(chunks=chunks_pl)
                    ds_combined = xr.merge([ds_normal, ds_special])
                else:
                    ds_combined = ds_normal

                ds_combined = ds_combined.chunk(chunks_pl).drop_vars("valid_time", errors="ignore")
                ds_combined = rename_and_enrich(
                    ds_combined,
                    has_ensemble=True,
                    init_times=group["datetimes"],
                )
                ds_combined = cast_float32(ds_combined)
                ds_combined = normalize_longitudes(ds_combined)
                ds_combined = normalize_latitudes(ds_combined)
                ds_combined = normalize_time_encodings(ds_combined)
                if debug_small:
                    log_ds_summary(f"ensemble.pl.chunk{i + 1}.{group['date']}", ds_combined)
                chunk_dsets.append(ds_combined)

            if not chunk_dsets:
                logging.error("No pressure-level datasets retrieved for ensemble chunk %d", i + 1)
                return False

            chunk_dataset = xr.concat(chunk_dsets, dim="init_time")
            chunk_dataset = chunk_dataset.sortby("init_time")

            logging.info(
                "Writing pressure level data to zarr (chunk %d/%d)...", i + 1, len(number_chunks)
            )
            ds_to_write = sanitize_dataset_attrs(chunk_dataset)
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
            output_exists = True

        # Add surface data (idempotent): only download/convert if needed
        write_surface = True
        if os.path.exists(output_file):
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                if (
                    "2t" in existing.data_vars
                    and "init_time" in existing.coords
                    and init_coord_matches(existing["init_time"], ordered_inits)
                ):
                    write_surface = False
                existing.close()
            except Exception:
                write_surface = True

        if write_surface:
            logging.info("Downloading surface data across %d day(s)...", len(date_groups))
            surface_dsets: list[xr.Dataset] = []
            for group in date_groups:
                hours_token = "/".join(group["hours"])
                request_sfc = {
                    "area": area,
                    "class": "od",
                    "date": group["date"],
                    "expver": "1",
                    "grid": grid,
                    "levtype": "sfc",
                    "number": "1/to/2/by/1" if debug_small else "1/to/50/by/1",
                    "param": single_level_params,
                    "step": "0/1"
                    if debug_small
                    else f"{interval}/to/{max_lead_hours}/by/{interval}",
                    "stream": "enfo",
                    "expect": "any",
                    "time": hours_token,
                    "type": "pf",
                }
                ds_single = earthkit.data.from_source("mars", request_sfc, lazily=True)
                ds_single = (
                    ds_single.to_xarray(chunks=chunks_surface)
                    .drop_vars("valid_time", errors="ignore")
                    .chunk(chunks_surface)
                )
                ds_single = rename_and_enrich(
                    ds_single,
                    has_ensemble=True,
                    init_times=group["datetimes"],
                )
                ds_single = cast_float32(ds_single)
                ds_single = normalize_longitudes(ds_single)
                ds_single = normalize_latitudes(ds_single)
                if debug_small:
                    log_ds_summary(f"ensemble.surface.{group['date']}", ds_single)
                surface_dsets.append(ds_single)

            if surface_dsets:
                ds_surface = xr.concat(surface_dsets, dim="init_time").sortby("init_time")
                ds_surface = normalize_time_encodings(ds_surface)
                ds_surface = normalize_longitudes(ds_surface)
                ds_surface = normalize_latitudes(ds_surface)
                ds_surface_sanitized = sanitize_dataset_attrs(
                    ds_surface.drop_vars(["z"], errors="ignore")
                )
                ds_surface_sanitized.to_zarr(
                    output_file,
                    consolidated=True,
                    zarr_format=2,
                    mode="a",
                )
            else:
                logging.warning(
                    "No surface datasets retrieved for ensemble request; skipping surface write"
                )
        else:
            logging.info("Surface data already present for all init_times; skipping surface write.")

        logging.info(f"Successfully downloaded ensemble data to {output_file}")
        return True

    except Exception as e:
        logging.error(f"Failed to download ensemble data for {init_summary}: {e}")
        return False


def download_ifs_control(
    output_dir, init_times, num_days, interval=6, *, fields, debug_small=False
):
    """Download IFS control data for one or more init_times."""

    if not init_times:
        logging.warning("No init_times requested for control download; skipping.")
        return True

    ordered_inits = dedupe_sorted_datetimes(init_times)
    date_str = ordered_inits[0].strftime("%Y%m%d%H%M")
    init_summary = (
        f"{ordered_inits[0]:%Y-%m-%d %H:%M} -> {ordered_inits[-1]:%Y-%m-%d %H:%M}"
        if len(ordered_inits) > 1
        else ordered_inits[0].strftime("%Y-%m-%d %H:%M")
    )
    expected_init_count = len(ordered_inits)
    date_groups = group_init_times_by_date(ordered_inits)
    if not date_groups:
        logging.warning("Unable to group init_times for control download; nothing to do.")
        return True

    cache_dir = os.path.join(output_dir, ".earthkit-cache")
    os.makedirs(cache_dir, exist_ok=True)
    settings.set("user-cache-directory", cache_dir)

    path = os.path.join(output_dir, date_str, MODEL_NAME)
    output_file = os.path.join(path, "ifs_control.zarr")

    if not os.path.exists(os.path.join(path, "fields.txt")):
        create_fields_file(path, fields)

    try:
        grid = fields["grid"]
        area = fields["area"]
        pressure_levels = fields["pressure_levels"]
        pressure_level_params = fields["pressure_level_params"]
        single_level_params = fields["single_level_params"]

        max_lead_hours = int(os.getenv("MAX_LEAD_TIME_HOURS", str(num_days * 24)))
        if max_lead_hours % interval != 0:
            max_lead_hours = (max_lead_hours // interval) * interval
        logging.info(
            "Downloading control data for %s (%d init_times, interval %dh, max lead %dh)",
            init_summary,
            expected_init_count,
            interval,
            max_lead_hours,
        )

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

        if os.path.exists(output_file):
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                has_all_init = "init_time" in existing.coords and init_coord_matches(
                    existing["init_time"], ordered_inits
                )
                has_dummy_ensemble = existing.sizes.get("ensemble", 0) == 1
                has_surface = "2t" in existing.data_vars
                existing.close()
                if has_all_init and has_dummy_ensemble and has_surface:
                    logging.info(
                        "Control store already complete for %s (%d init_times); skipping.",
                        init_summary,
                        expected_init_count,
                    )
                    return True
            except Exception as e:
                logging.warning(
                    "Failed to inspect existing control store for completeness: %s. Will overwrite.",
                    e,
                )

        daily_pl: list[xr.Dataset] = []
        for group in date_groups:
            hours_token = "/".join(group["hours"])
            request_pl = {
                "area": area,
                "class": "od",
                "date": group["date"],
                "expver": "1",
                "grid": grid,
                "levtype": "pl",
                "levelist": pressure_levels,
                "param": pressure_level_params,
                "step": "0/1" if debug_small else f"{interval}/to/{max_lead_hours}/by/{interval}",
                "stream": "enfo",
                "expect": "any",
                "time": hours_token,
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
            ds_combined = rename_and_enrich(
                ds_combined,
                has_ensemble=False,
                init_times=group["datetimes"],
                add_dummy_ensemble=True,
            )
            ds_combined = cast_float32(ds_combined)
            ds_combined = normalize_longitudes(ds_combined)
            ds_combined = normalize_latitudes(ds_combined)
            ds_combined = normalize_time_encodings(ds_combined)
            if debug_small:
                log_ds_summary(f"control.pl.{group['date']}", ds_combined)
            daily_pl.append(ds_combined)

        if not daily_pl:
            logging.error("No pressure-level datasets retrieved for control run")
            return False

        ds_pressure_all = xr.concat(daily_pl, dim="init_time").sortby("init_time")
        ds_pressure_all = sanitize_dataset_attrs(ds_pressure_all)
        if os.path.exists(output_file):
            import shutil

            shutil.rmtree(output_file, ignore_errors=True)
        ds_pressure_all.to_zarr(output_file, consolidated=True, zarr_format=2, mode="w")

        surface_sets: list[xr.Dataset] = []
        for group in date_groups:
            hours_token = "/".join(group["hours"])
            request_sfc = {
                "area": area,
                "class": "od",
                "date": group["date"],
                "expver": "1",
                "grid": grid,
                "levtype": "sfc",
                "param": single_level_params,
                "step": "0/1" if debug_small else f"{interval}/to/{max_lead_hours}/by/{interval}",
                "stream": "enfo",
                "expect": "any",
                "time": hours_token,
                "type": "cf",
            }
            ds_single = earthkit.data.from_source("mars", request_sfc, lazily=True)
            ds_single = (
                ds_single.to_xarray(chunks=chunks_surface)
                .drop_vars("valid_time", errors="ignore")
                .chunk(chunks_surface)
            )
            ds_single = rename_and_enrich(
                ds_single,
                has_ensemble=False,
                init_times=group["datetimes"],
                add_dummy_ensemble=True,
            )
            ds_single = cast_float32(ds_single)
            ds_single = normalize_longitudes(ds_single)
            ds_single = normalize_latitudes(ds_single)
            if debug_small:
                log_ds_summary(f"control.surface.{group['date']}", ds_single)
            surface_sets.append(ds_single)

        if surface_sets:
            ds_surface = xr.concat(surface_sets, dim="init_time").sortby("init_time")
            ds_surface = normalize_time_encodings(ds_surface)
            ds_surface = normalize_longitudes(ds_surface)
            ds_surface = normalize_latitudes(ds_surface)
            ds_surface_sanitized = sanitize_dataset_attrs(
                ds_surface.drop_vars(["z"], errors="ignore")
            )
            ds_surface_sanitized.to_zarr(output_file, consolidated=True, zarr_format=2, mode="a")
        else:
            logging.warning(
                "No surface datasets retrieved for control run; continuing without surface fields"
            )

        logging.info(f"Successfully downloaded control data to {output_file}")
        return True

    except Exception as e:
        logging.error(f"Failed to download control data for {init_summary}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download IFS data for a single date")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("date_time", type=str, help="Date and time in format YYYYMMDDHHMM")
    parser.add_argument("num_days", type=int, help="Number of days to download")
    # Model name is fixed to ESFM; no CLI option needed
    parser.add_argument("--interval", type=int, default=6, help="Time step in hours")
    parser.add_argument(
        "--range-end",
        type=str,
        help="Inclusive end datetime (YYYYMMDDHHMM) for multi-init downloads",
    )
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

    # Determine range end (optional) and validate interval alignment
    interval_minutes = args.interval * 60
    start_minutes = date_time.hour * 60 + date_time.minute
    if start_minutes % interval_minutes != 0:
        logging.error(
            "Start time %s is not aligned with interval %dh",
            date_time.strftime("%Y-%m-%d %H:%M"),
            args.interval,
        )
        return 1

    range_end_dt = None
    if args.range_end:
        try:
            range_end_dt = datetime.strptime(args.range_end, "%Y%m%d%H%M")
        except ValueError:
            logging.error(f"Invalid range-end format: {args.range_end}. Use YYYYMMDDHHMM")
            return 1
        if range_end_dt < date_time:
            logging.error("range_end must be >= start date")
            return 1
        end_minutes = range_end_dt.hour * 60 + range_end_dt.minute
        remainder = end_minutes % interval_minutes
        if remainder != 0:
            aligned = range_end_dt - timedelta(minutes=remainder)
            logging.info(
                "Aligning range_end from %s to %s to respect %dh interval",
                range_end_dt.strftime("%Y-%m-%d %H:%M"),
                aligned.strftime("%Y-%m-%d %H:%M"),
                args.interval,
            )
            range_end_dt = aligned
        if range_end_dt < date_time:
            logging.error("range_end became earlier than start after alignment; aborting")
            return 1

    init_times = generate_init_times(date_time, range_end_dt, args.interval)
    logging.info("Total init_times to fetch: %d (interval %dh)", len(init_times), args.interval)

    # Choose fields for this run
    fields = get_debug_fields() if args.debug_small else ESFM_FIELDS

    success = True

    # Download ensemble data
    if args.download_type in ["ensemble", "both"]:
        success &= download_ifs_ensemble(
            args.output_dir,
            init_times,
            args.num_days,
            args.interval,
            fields=fields,
            debug_small=args.debug_small,
        )

    # Download control data
    if args.download_type in ["control", "both"]:
        success &= download_ifs_control(
            args.output_dir,
            init_times,
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
