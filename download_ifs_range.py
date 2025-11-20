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
from typing import Any, Callable, Iterable, Mapping, Sequence, Set

import earthkit.data
import numpy as np
import xarray as xr
from earthkit.data import settings

# Model name can be overridden via environment variable MODEL_NAME (default: esfm)
MODEL_NAME = os.getenv("MODEL_NAME", "esfm")


FIELD_ENV_SPECS: tuple[tuple[str, str, Callable[[str], Any], int | None], ...] = (
    ("grid", "IFS_GRID", float, 2),
    ("area", "IFS_AREA", float, 4),
    ("pressure_levels", "IFS_PRESSURE_LEVELS", int, None),
    ("pressure_level_params", "IFS_PRESSURE_LEVEL_PARAMS", str, None),
    ("single_level_params", "IFS_SINGLE_LEVEL_PARAMS", str, None),
)


def _parse_env_list(env_name: str, caster: Callable[[str], Any]) -> list[Any] | None:
    """Parse a comma-separated environment variable using the provided caster."""

    raw = os.getenv(env_name)
    if raw is None or raw.strip() == "":
        return None
    values: list[Any] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            values.append(caster(stripped))
        except ValueError as exc:  # pragma: no cover - basic config validation
            raise ValueError(f"{env_name} entry '{stripped}' is invalid: {exc}") from exc
    if not values:
        raise ValueError(f"{env_name} did not contain any usable values")
    return values


def load_fields_from_env() -> dict[str, list[Any]]:
    """Load field selection from environment variables (raises if any are missing)."""

    fields: dict[str, list[Any]] = {}
    missing: list[str] = []
    for key, env_name, caster, expected_len in FIELD_ENV_SPECS:
        parsed = _parse_env_list(env_name, caster)
        if parsed is None:
            missing.append(env_name)
            continue
        if expected_len is not None and len(parsed) != expected_len:
            raise ValueError(
                f"{env_name} must provide exactly {expected_len} comma-separated values"
            )
        fields[key] = parsed

    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing field configuration via environment variables: {missing_str}. "
            "Set these in config.env before running."
        )

    return fields


def get_debug_fields():
    """Return a greatly reduced set of fields for fast debug runs."""
    return {
        "grid": [1.0, 1.0],  # coarser grid
        "area": [60, -30, 30, 30],  # small region (N/W/S/E)
        "pressure_levels": [1000, 925],  # single pressure level
        "pressure_level_params": ["u"],  # single PL variable
        "single_level_params": ["2t"],  # single SL variable
    }


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
            force=True,
        )
    else:
        logging.basicConfig(level=level, format=format_str, force=True)


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


def append_init_to_store(output_file: str, ds: xr.Dataset) -> None:
    """Persist a single-init dataset by appending along init_time."""

    exists = os.path.exists(output_file)
    mode = "a" if exists else "w"
    sanitized = sanitize_dataset_attrs(ds)
    write_kwargs = {
        "consolidated": True,
        "zarr_format": 2,
        "mode": mode,
    }
    if exists:
        write_kwargs["append_dim"] = "init_time"
    sanitized.to_zarr(output_file, **write_kwargs)


def load_existing_init_times(zarr_path: str) -> Set[np.datetime64]:
    """Return the set of init_time values already stored in a Zarr archive."""

    existing: Set[np.datetime64] = set()
    if not os.path.exists(zarr_path):
        return existing
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        if "init_time" in ds.coords:
            values = ds["init_time"].values.astype("datetime64[ns]")
            existing = set(values.tolist())
        ds.close()
    except Exception as exc:
        logging.warning("Failed to inspect existing store %s: %s", zarr_path, exc)
    return existing


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

    ds = ds.drop_vars(["time"], errors="ignore")
    if "init_time" in ds.coords and "init_time" not in ds.dims:
        ds = ds.drop_vars(["init_time"], errors="ignore")

    ts = to_datetime64(init_times) if init_times else None
    if "init_time" not in ds.dims:
        if ts is None:
            raise ValueError("init_times must be provided when dataset lacks init_time coord")
        ds = ds.expand_dims(init_time=ts)
    if ts is not None:
        if ds.sizes.get("init_time", len(ts)) != len(ts):
            raise ValueError("init_time dimension mismatch")
        ds = ds.assign_coords(init_time=("init_time", ts))
    elif "init_time" in ds.dims:
        ds = ds.assign_coords(init_time=ds["init_time"].astype("datetime64[ns]"))
    else:
        raise ValueError("init_times must be provided when dataset lacks init_time dim")

    ds["init_time"] = ds["init_time"].astype("datetime64[ns]")
    ds["init_time"].attrs = {}

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
            raw = lt.values
            if np.issubdtype(raw.dtype, np.timedelta64):
                ints = raw.astype("int64")
                # Genuine ns values are O(1e14); smaller numbers mean the array still
                # stores lead times as bare hours, so rescale explicitly.
                if ints.size and np.nanmax(np.abs(ints)) < 10**10:
                    td = (ints * np.timedelta64(1, "h")).astype("timedelta64[ns]")
                else:
                    td = raw.astype("timedelta64[ns]")
            else:
                # Treat numeric lead_time as hours by convention (IFS step is in hours)
                ints = raw.astype("int64")
                td = (ints * np.timedelta64(1, "h")).astype("timedelta64[ns]")
            ds = ds.assign_coords(lead_time=("lead_time", td))
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
    requested_init_set = {np.datetime64(dt, "ns") for dt in ordered_inits}
    if not ordered_inits:
        logging.warning("Unable to prepare init_times for ensemble download; nothing to do.")
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
        existing_init_times = load_existing_init_times(output_file)
        if output_exists:
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                existing_count = int(existing.sizes.get("ensemble", 0))
                has_surface_data = (
                    "2m_temperature" in existing.data_vars or "2t" in existing.data_vars
                )
                existing.close()
                if (
                    requested_init_set.issubset(existing_init_times)
                    and existing_count >= expected_ensemble
                    and has_surface_data
                ):
                    logging.info(
                        "Existing ensemble store already complete (%d init_times). Skipping.",
                        len(existing_init_times),
                    )
                    return True
                if existing_count < expected_ensemble:
                    logging.warning(
                        "Existing ensemble dimension %d smaller than expected %d; new data will be appended.",
                        existing_count,
                        expected_ensemble,
                    )
                if not has_surface_data:
                    logging.warning(
                        "Existing ensemble store lacks surface vars; new runs will replenish them."
                    )
            except Exception as e:
                logging.warning(
                    "Failed to inspect existing ensemble store for incremental writes: %s", e
                )
                existing_init_times = set()

        step_ranges = ["0/1"] if debug_small else [f"{interval}/to/{max_lead_hours}/by/{interval}"]
        number_token = "1/to/2/by/1" if debug_small else "1/to/50/by/1"
        total_inits = len(ordered_inits)

        for idx, init_dt in enumerate(ordered_inits, start=1):
            init_key = np.datetime64(init_dt, "ns")
            if init_key in existing_init_times:
                logging.info(
                    "Ensemble init %s already persisted; skipping (%d/%d).",
                    init_dt.strftime("%Y-%m-%d %H:%M"),
                    idx,
                    total_inits,
                )
                continue

            date_token = init_dt.strftime("%Y-%m-%d")
            hour_token = init_dt.strftime("%H")

            init_step_sets: list[xr.Dataset] = []
            for step_expr in step_ranges:
                request_pl = {
                    "area": area,
                    "class": "od",
                    "date": date_token,
                    "expver": "1",
                    "grid": grid,
                    "levtype": "pl",
                    "levelist": pressure_levels,
                    "param": pressure_level_params,
                    "number": number_token,
                    "step": step_expr,
                    "stream": "enfo",
                    "expect": "any",
                    "time": hour_token,
                    "type": "pf",
                }

                logging.info(
                    "Submitting ensemble pressure request for init %s step %s",
                    init_dt.strftime("%Y-%m-%d %H:%M"),
                    step_expr,
                )

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
                    has_ensemble=True,
                    init_times=[init_dt],
                )
                ds_combined = cast_float32(ds_combined)
                ds_combined = normalize_longitudes(ds_combined)
                ds_combined = normalize_latitudes(ds_combined)
                ds_combined = normalize_time_encodings(ds_combined)
                if debug_small:
                    log_ds_summary(f"ensemble.pl.{init_dt:%Y%m%d%H}.{step_expr}", ds_combined)
                init_step_sets.append(ds_combined)

            if not init_step_sets:
                logging.error("No pressure data retrieved for ensemble init %s", init_dt)
                return False

            init_pl = (
                xr.concat(init_step_sets, dim="lead_time").sortby("lead_time")
                if len(init_step_sets) > 1
                else init_step_sets[0]
            )
            logging.info(
                "Ensemble pressure complete %d/%d for init %s",
                idx,
                total_inits,
                init_dt.strftime("%Y-%m-%d %H:%M"),
            )

            init_surface_sets: list[xr.Dataset] = []
            for step_expr in step_ranges:
                request_sfc = {
                    "area": area,
                    "class": "od",
                    "date": date_token,
                    "expver": "1",
                    "grid": grid,
                    "levtype": "sfc",
                    "number": number_token,
                    "param": single_level_params,
                    "step": step_expr,
                    "stream": "enfo",
                    "expect": "any",
                    "time": hour_token,
                    "type": "pf",
                }
                logging.info(
                    "Submitting ensemble surface request for init %s step %s",
                    init_dt.strftime("%Y-%m-%d %H:%M"),
                    step_expr,
                )
                ds_single = earthkit.data.from_source("mars", request_sfc, lazily=True)
                ds_single = (
                    ds_single.to_xarray(chunks=chunks_surface)
                    .drop_vars("valid_time", errors="ignore")
                    .chunk(chunks_surface)
                )
                ds_single = rename_and_enrich(
                    ds_single,
                    has_ensemble=True,
                    init_times=[init_dt],
                )
                ds_single = cast_float32(ds_single)
                ds_single = normalize_longitudes(ds_single)
                ds_single = normalize_latitudes(ds_single)
                if debug_small:
                    log_ds_summary(f"ensemble.surface.{init_dt:%Y%m%d%H}.{step_expr}", ds_single)
                init_surface_sets.append(ds_single)

            ds_surface = None
            if init_surface_sets:
                ds_surface = (
                    xr.concat(init_surface_sets, dim="lead_time").sortby("lead_time")
                    if len(init_surface_sets) > 1
                    else init_surface_sets[0]
                )
                ds_surface = normalize_time_encodings(ds_surface)
                ds_surface = normalize_longitudes(ds_surface)
                ds_surface = normalize_latitudes(ds_surface)
                logging.info(
                    "Ensemble surface complete %d/%d for init %s",
                    idx,
                    total_inits,
                    init_dt.strftime("%Y-%m-%d %H:%M"),
                )
            else:
                logging.warning(
                    "No surface datasets retrieved for ensemble init %s; continuing without surface fields",
                    init_dt.strftime("%Y-%m-%d %H:%M"),
                )

            ds_to_write = (
                xr.merge([init_pl, ds_surface], compat="no_conflicts")
                if ds_surface is not None
                else init_pl
            )
            append_init_to_store(output_file, ds_to_write)
            existing_init_times.add(init_key)
            logging.info(
                "Persisted ensemble init %s (%d/%d) to %s",
                init_dt.strftime("%Y-%m-%d %H:%M"),
                idx,
                total_inits,
                output_file,
            )

        missing_inits = requested_init_set.difference(existing_init_times)
        if missing_inits:
            logging.error(
                "Ensemble download finished but %d init_times are still missing: %s",
                len(missing_inits),
                sorted(str(val) for val in list(missing_inits))[:5],
            )
            return False

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
    requested_init_set = {np.datetime64(dt, "ns") for dt in ordered_inits}
    if not ordered_inits:
        logging.warning("Unable to prepare init_times for control download; nothing to do.")
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

        request_step = "0/1" if debug_small else f"{interval}/to/{max_lead_hours}/by/{interval}"
        existing_init_times = load_existing_init_times(output_file)
        if os.path.exists(output_file):
            try:
                existing = xr.open_zarr(output_file, consolidated=True)
                has_surface = "2m_temperature" in existing.data_vars or "2t" in existing.data_vars
                existing.close()
                if requested_init_set.issubset(existing_init_times) and has_surface:
                    logging.info(
                        "Control store already complete for %s (%d init_times); skipping.",
                        init_summary,
                        expected_init_count,
                    )
                    return True
                if not has_surface:
                    logging.warning(
                        "Existing control store missing surface vars; new downloads will append them."
                    )
            except Exception as e:
                logging.warning(
                    "Failed to inspect existing control store for incremental writes: %s", e
                )
                existing_init_times = set()

        total_inits = len(ordered_inits)
        for idx, init_dt in enumerate(ordered_inits, start=1):
            init_key = np.datetime64(init_dt, "ns")
            if init_key in existing_init_times:
                logging.info(
                    "Control init %s already persisted; skipping (%d/%d).",
                    init_dt.strftime("%Y-%m-%d %H:%M"),
                    idx,
                    total_inits,
                )
                continue

            date_token = init_dt.strftime("%Y-%m-%d")
            hour_token = init_dt.strftime("%H")
            request_pl = {
                "area": area,
                "class": "od",
                "date": date_token,
                "expver": "1",
                "grid": grid,
                "levtype": "pl",
                "levelist": pressure_levels,
                "param": pressure_level_params,
                "step": request_step,
                "stream": "enfo",
                "expect": "any",
                "time": hour_token,
                "type": "cf",
            }

            logging.info(
                "Submitting control pressure request for init %s",
                init_dt.strftime("%Y-%m-%d %H:%M"),
            )

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
                init_times=[init_dt],
                add_dummy_ensemble=True,
            )
            ds_combined = cast_float32(ds_combined)
            ds_combined = normalize_longitudes(ds_combined)
            ds_combined = normalize_latitudes(ds_combined)
            ds_combined = normalize_time_encodings(ds_combined)
            if debug_small:
                log_ds_summary(f"control.pl.{init_dt:%Y%m%d%H}", ds_combined)
            logging.info(
                "Control pressure complete %d/%d for init %s",
                idx,
                total_inits,
                init_dt.strftime("%Y-%m-%d %H:%M"),
            )

            request_sfc = {
                "area": area,
                "class": "od",
                "date": date_token,
                "expver": "1",
                "grid": grid,
                "levtype": "sfc",
                "param": single_level_params,
                "step": request_step,
                "stream": "enfo",
                "expect": "any",
                "time": hour_token,
                "type": "cf",
            }
            logging.info(
                "Submitting control surface request for init %s",
                init_dt.strftime("%Y-%m-%d %H:%M"),
            )
            ds_single = earthkit.data.from_source("mars", request_sfc, lazily=True)
            ds_single = (
                ds_single.to_xarray(chunks=chunks_surface)
                .drop_vars("valid_time", errors="ignore")
                .chunk(chunks_surface)
            )
            ds_single = rename_and_enrich(
                ds_single,
                has_ensemble=False,
                init_times=[init_dt],
                add_dummy_ensemble=True,
            )
            ds_single = cast_float32(ds_single)
            ds_single = normalize_longitudes(ds_single)
            ds_single = normalize_latitudes(ds_single)
            if debug_small:
                log_ds_summary(f"control.surface.{init_dt:%Y%m%d%H}", ds_single)
            ds_single = normalize_time_encodings(ds_single)
            logging.info(
                "Control surface complete %d/%d for init %s",
                idx,
                total_inits,
                init_dt.strftime("%Y-%m-%d %H:%M"),
            )

            ds_to_write = xr.merge([ds_combined, ds_single], compat="no_conflicts")
            append_init_to_store(output_file, ds_to_write)
            existing_init_times.add(init_key)
            logging.info(
                "Persisted control init %s (%d/%d) to %s",
                init_dt.strftime("%Y-%m-%d %H:%M"),
                idx,
                total_inits,
                output_file,
            )

        missing_inits = requested_init_set.difference(existing_init_times)
        if missing_inits:
            logging.error(
                "Control download finished but %d init_times are still missing: %s",
                len(missing_inits),
                sorted(str(val) for val in list(missing_inits))[:5],
            )
            return False

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
    if args.debug_small:
        fields = get_debug_fields()
    else:
        try:
            fields = load_fields_from_env()
        except ValueError as exc:
            logging.error("Invalid field configuration: %s", exc)
            return 1

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
