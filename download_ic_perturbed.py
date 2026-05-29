#!/usr/bin/env python3
"""Download IFS ENS perturbed analyses (type=pf, step=0) for AI model IC use.

What this downloads
-------------------
Per (date, time) -- one tape mount per request -- the script issues 2 MARS requests:

1. Upper-air on pressure levels (PL), 13 levels including 600 hPa:
   levelist = 50/100/150/200/250/300/400/500/600/700/850/925/1000
   param    = T, U, V, Q, Z, W (130/131/132/133/129/135)

2. Surface fields (SFC):
   param    = MSL, 2T, 10U, 10V, 100U, 100V, SP, TCWV (151/167/165/166/246/247/134/137)

Both requests use:
   class=od, stream=enfo, type=pf, step=0, number=1..50

Why type=pf step=0 (not icp): PF step=0 IS the per-member perturbed analysis ready
to feed the model. ICP would require us to add the control analysis back per
member -- simpler to grab the pre-summed product.

Output
------
Single Zarr v3 store with sharding codec at
$OUT_ZARR (default /capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr)

Schema:
  dims = (ensemble=50, init_time=N, [level=13 for 3D vars], latitude=721, longitude=1440)
  inner chunks = (50, 1, 1, 721, 1440) for 3D, (50, 1, 721, 1440) for 2D
  shards       = one shard per variable (covers full array)
  no lead_time dim (this is analysis, not forecast)

Idempotency: skip (init_time) cells that already have valid data on disk.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import earthkit.data
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Constants: variable + level lists for the AI baseline models (union of
# Aurora, GraphCast, SFNO inference logs, 2026-05-28).
# ----------------------------------------------------------------------------
PL_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# MARS param IDs -> ECMWF GRIB shortName -> what AI models call them
PL_PARAMS = {
    "130": "t",   # Temperature
    "131": "u",   # U-component of wind
    "132": "v",   # V-component of wind
    "133": "q",   # Specific humidity
    "129": "z",   # Geopotential
    "135": "w",   # Vertical velocity (omega, Pa/s)
}

SFC_PARAMS = {
    "151": "msl",   # Mean sea level pressure
    "167": "2t",    # 2m temperature
    "165": "10u",   # 10m u-component wind
    "166": "10v",   # 10m v-component wind
    "246": "100u",  # 100m u-component wind (SFNO only)
    "247": "100v",  # 100m v-component wind (SFNO only)
    "134": "sp",    # Surface pressure (SFNO only)
    "137": "tcwv",  # Total column water vapour (SFNO only)
}

N_MEMBERS = 50            # 50 perturbed members (no control)
GRID = [0.25, 0.25]
AREA = [90.0, -180.0, -90.0, 180.0]
LAT_N, LON_N = 721, 1440

# ----------------------------------------------------------------------------
# Init time generation: 8 weeks of baseline + T-6h extension for GraphCast/Aurora
# ----------------------------------------------------------------------------
WEEK_STARTS = [
    "2023-01-02", "2023-04-02", "2023-07-02", "2023-10-02",
    "2024-01-02", "2024-04-02", "2024-07-02", "2024-10-02",
]


def build_init_times() -> list[datetime]:
    """Return all 224 init times: 8 weeks, each 28 6-hourly samples spanning
    [week_start_day_0 18:00, week_start_day_6 12:00]."""
    out: list[datetime] = []
    for ws in WEEK_STARTS:
        d0 = datetime.fromisoformat(ws)
        start = d0 - timedelta(hours=6)   # T-6h of first baseline init (00Z)
        end = d0 + timedelta(days=6, hours=12)   # last baseline init (12Z on day 6)
        cur = start
        while cur <= end:
            out.append(cur)
            cur += timedelta(hours=6)
    return out


# ----------------------------------------------------------------------------
# Zarr v3 sharded store setup -- match SwissClim DESIRED_CHUNKS pattern
# ----------------------------------------------------------------------------
def _make_inner_chunks(dim_names: tuple[str, ...], dim_sizes: dict[str, int]) -> tuple[int, ...]:
    """Inner chunk size per dim. Matches SwissClim DESIRED_CHUNKS:
       ensemble=full, init_time=1, level=1, lat=full, lon=full."""
    out = []
    for d in dim_names:
        if d == "ensemble":
            out.append(dim_sizes[d])
        elif d in ("init_time", "lead_time", "level"):
            out.append(1)
        elif d in ("latitude", "longitude"):
            out.append(dim_sizes[d])
        else:
            out.append(dim_sizes[d])
    return tuple(out)


def init_zarr_store(
    out_path: Path,
    init_times: list[datetime],
) -> None:
    """Initialise an empty Zarr v3 sharded store with the full schema, lazy fill
    values. Subsequent calls write each init_time slice via region write."""
    if out_path.exists():
        logging.info("Zarr store already exists at %s -- skipping init", out_path)
        return

    n_init = len(init_times)
    coords = {
        "ensemble": np.arange(1, N_MEMBERS + 1, dtype="int32"),
        "init_time": np.array(init_times, dtype="datetime64[ns]"),
        "level": np.array(PL_LEVELS, dtype="int32"),
        "latitude": np.linspace(90.0, -90.0, LAT_N, dtype="float32"),
        "longitude": np.linspace(0.0, 359.75, LON_N, dtype="float32"),
    }
    ds = xr.Dataset(coords=coords)

    # 3D vars: (ensemble, init_time, level, latitude, longitude)
    dims_3d = ("ensemble", "init_time", "level", "latitude", "longitude")
    shape_3d = (N_MEMBERS, n_init, len(PL_LEVELS), LAT_N, LON_N)
    inner_3d = _make_inner_chunks(dims_3d, dict(zip(dims_3d, shape_3d)))

    # 2D vars: (ensemble, init_time, latitude, longitude)
    dims_2d = ("ensemble", "init_time", "latitude", "longitude")
    shape_2d = (N_MEMBERS, n_init, LAT_N, LON_N)
    inner_2d = _make_inner_chunks(dims_2d, dict(zip(dims_2d, shape_2d)))

    encoding: dict[str, dict] = {}
    for short in PL_PARAMS.values():
        ds[short] = (dims_3d, np.empty(shape_3d, dtype="float32"))
        encoding[short] = {"chunks": inner_3d, "shards": shape_3d}
    for short in SFC_PARAMS.values():
        ds[short] = (dims_2d, np.empty(shape_2d, dtype="float32"))
        encoding[short] = {"chunks": inner_2d, "shards": shape_2d}

    # Coord encodings (small dims, default chunks)
    for c in ("ensemble", "init_time", "level", "latitude", "longitude"):
        encoding[c] = {"chunks": (ds.sizes[c],)}

    logging.info("Creating Zarr v3 sharded store at %s", out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Lazy fill (no real bytes written for chunks not touched).
    ds.to_zarr(out_path, mode="w-", encoding=encoding, consolidated=False, zarr_format=3,
               compute=False)


# ----------------------------------------------------------------------------
# MARS request + write per init_time
# ----------------------------------------------------------------------------
# Levels NOT archived by MARS for type=pf step=0 on the PL stream (we asked
# for 13, MARS returns 11). We interpolate them in log-pressure space after
# download from the two adjacent archived levels.
INTERP_LEVELS = {
    150: (100, 200),   # f(150) = 0.42*f(100) + 0.58*f(200)
    600: (500, 700),   # f(600) = 0.46*f(500) + 0.54*f(700)
}

# Earthkit cache root -- where MARS GRIBs land before earthkit reads them.
# Pinned to /iopsstor/scratch so the login node disk never accumulates GRIB
# spill. The cleanup helper below removes files after each successful init,
# but this is the belt-and-suspenders location.
EARTHKIT_CACHE = Path("/iopsstor/scratch/cscs/sadamov/earthkit_cache")


def _configure_earthkit_cache() -> None:
    """Force the earthkit-data cache onto /iopsstor/scratch and set it BEFORE
    any earthkit MARS call. Without this, the default user-cache-directory
    from the user's config file (currently /capstor/...) would be used and
    every PL request would write 6+ GB to capstor before we get a chance
    to clean it up."""
    EARTHKIT_CACHE.mkdir(parents=True, exist_ok=True)
    import earthkit.data as ed
    ed.config.set("cache-policy", "user")
    ed.config.set("user-cache-directory", str(EARTHKIT_CACHE))
    ed.config.set("temporary-cache-directory-root", str(EARTHKIT_CACHE))
    ed.config.set("temporary-directory-root", str(EARTHKIT_CACHE))
    # Two caps: an absolute size limit AND a percent-of-disk floor.
    # maximum-cache-size accepts absolute units (G, M, etc); maximum-cache-disk-usage
    # is percent-only. Keep cache well below 20 GB -- ONE init's PL+SFC GRIB is
    # ~6 GB so this leaves comfortable headroom and forces cleanup quickly.
    ed.config.set("maximum-cache-size", "20G")
    ed.config.set("maximum-cache-disk-usage", "95%")
    logging.info("earthkit cache: %s", ed.config.get("user-cache-directory"))


def already_written(out_path: Path, init_dt: datetime) -> bool:
    """Check if this init slice was fully written. Samples (member, level, var)
    triples to catch partial writes that left NaN in some chunks."""
    try:
        ds = xr.open_zarr(out_path, consolidated=False)
        init_idx = np.where(ds.init_time.values == np.datetime64(init_dt, "ns"))[0]
        if len(init_idx) == 0:
            return False
        i = int(init_idx[0])
        # Spot-check 3 var x 2 member x 2 level combinations
        for v in ("t", "u", "z"):
            for m in (0, 25):
                for L in (0, 5):
                    arr = ds[v].isel(init_time=i, ensemble=m, level=L).values
                    if not np.all(np.isfinite(arr)):
                        return False
        # And surface var
        for v in ("msl", "2t"):
            for m in (0, 25):
                arr = ds[v].isel(init_time=i, ensemble=m).values
                if not np.all(np.isfinite(arr)):
                    return False
        return True
    except Exception:
        return False


def _interpolate_missing_levels(ds: xr.Dataset) -> xr.Dataset:
    """Inject missing 150 + 600 hPa PL via linear-in-log-p interpolation
    from the adjacent archived levels."""
    if "level" not in ds.dims:
        return ds
    present = set(int(L) for L in ds.level.values)
    additions: list[xr.Dataset] = []
    for new_L, (lo, hi) in INTERP_LEVELS.items():
        if new_L in present:
            continue
        if lo not in present or hi not in present:
            logging.warning("Cannot interpolate L=%d: missing %d or %d", new_L, lo, hi)
            continue
        log_lo, log_hi = np.log(lo), np.log(hi)
        alpha = (np.log(new_L) - log_lo) / (log_hi - log_lo)
        slab = (1 - alpha) * ds.sel(level=lo) + alpha * ds.sel(level=hi)
        slab = slab.expand_dims({"level": [new_L]}).astype("float32")
        additions.append(slab)
    if additions:
        ds = xr.concat([ds, *additions], dim="level").sortby("level", ascending=False)
    return ds


def _cleanup_earthkit_cache() -> None:
    """Remove all mars-retriever-*.cache* files from the earthkit cache so the
    next init starts with a clean disk. Cumulative cache would otherwise hit
    ~1.3 TB across 224 inits."""
    if not EARTHKIT_CACHE.exists():
        return
    for f in EARTHKIT_CACHE.glob("mars-retriever-*.cache*"):
        try:
            f.unlink()
        except Exception:
            pass


def fetch_and_write_one(
    out_path: Path,
    init_dt: datetime,
    init_idx: int,
) -> bool:
    """Fetch PL + SFC for one init_time from MARS, write to Zarr slice."""
    if already_written(out_path, init_dt):
        logging.info("[%s] already written -- skipping", init_dt)
        return True

    date_token = init_dt.strftime("%Y-%m-%d")
    hour_token = init_dt.strftime("%H")
    member_token = "/".join(str(i) for i in range(1, N_MEMBERS + 1))

    common = {
        "area": "/".join(str(x) for x in AREA),
        "class": "od",
        "date": date_token,
        "expver": "1",
        "expect": "any",   # tolerate missing param/level combos (some PL not archived for type=pf step=0)
        "grid": "/".join(str(x) for x in GRID),
        "number": member_token,
        "step": "0",
        "stream": "enfo",
        "time": hour_token,
        "type": "pf",
    }

    # Pressure level request
    pl_req = {
        **common,
        "levtype": "pl",
        "levelist": "/".join(str(L) for L in PL_LEVELS),
        "param": "/".join(PL_PARAMS.keys()),
    }
    logging.info("[%s] MARS PL request: %s", init_dt, pl_req)
    pl_src = earthkit.data.from_source("mars", pl_req, lazily=True)
    pl_ds = pl_src.to_xarray(chunks={"number": -1, "step": 1, "level": 1})

    # Surface request
    sfc_req = {
        **common,
        "levtype": "sfc",
        "param": "/".join(SFC_PARAMS.keys()),
    }
    logging.info("[%s] MARS SFC request: %s", init_dt, sfc_req)
    sfc_src = earthkit.data.from_source("mars", sfc_req, lazily=True)
    sfc_ds = sfc_src.to_xarray(chunks={"number": -1, "step": 1})

    # Normalise: rename dims/coords, drop unused, align with our schema
    pl_ds = _normalise(pl_ds, has_level=True)
    sfc_ds = _normalise(sfc_ds, has_level=False)

    # Interpolate 150 + 600 hPa from neighbours (log-pressure linear)
    pl_ds = _interpolate_missing_levels(pl_ds)

    # Combine & write
    merged = xr.merge([pl_ds, sfc_ds], compat="override")
    merged = merged.expand_dims({"init_time": [np.datetime64(init_dt, "ns")]})

    logging.info("[%s] writing slice idx=%d to %s", init_dt, init_idx, out_path)
    merged.to_zarr(out_path, region={"init_time": slice(init_idx, init_idx + 1)},
                   consolidated=False)

    # Free the GRIB cache so we don't accumulate ~6 GB per init
    _cleanup_earthkit_cache()
    return True


def _normalise(ds: xr.Dataset, has_level: bool) -> xr.Dataset:
    """Rename earthkit MARS-style dims to our schema."""
    rename = {}
    if "number" in ds.dims:
        rename["number"] = "ensemble"
    if "isobaricInhPa" in ds.dims:
        rename["isobaricInhPa"] = "level"
    if "lat" in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    if "valid_time" in ds.coords:
        ds = ds.drop_vars("valid_time", errors="ignore")
    if "step" in ds.coords:
        ds = ds.drop_vars("step", errors="ignore")
    if "time" in ds.coords:
        ds = ds.drop_vars("time", errors="ignore")
    # Cast to float32, normalise longitudes 0..360, latitudes 90..-90
    for v in ds.data_vars:
        ds[v] = ds[v].astype("float32")
    if "longitude" in ds.coords and float(ds.longitude.min()) < 0:
        ds = ds.assign_coords(longitude=(ds.longitude % 360))
        ds = ds.sortby("longitude")
    if "latitude" in ds.coords and float(ds.latitude[0]) < float(ds.latitude[-1]):
        ds = ds.sortby("latitude", ascending=False)
    return ds


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Download IFS ENS IC (type=pf step=0)")
    parser.add_argument(
        "--out-zarr",
        default="/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr",
        help="Output Zarr v3 store path",
    )
    parser.add_argument(
        "--inits", type=str, default="",
        help="Comma-separated ISO init_times to fetch (default: all 224)",
    )
    parser.add_argument(
        "--skip-init-store",
        action="store_true",
        help="Don't create the empty Zarr store (assume it exists)",
    )
    parser.add_argument(
        "--single-init",
        action="store_true",
        help="Paranoia mode: download only the FIRST init_time, do not "
             "create or write to the zarr store. Print summary of fetched "
             "fields for inspection. Use this before the full 224-init run.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # FORCE the earthkit cache to /iopsstor/scratch BEFORE any MARS call.
    # Login-node safety: prevents 6 GB per request from landing on capstor.
    _configure_earthkit_cache()

    all_inits = build_init_times()
    logging.info("Total init_times to consider: %d", len(all_inits))

    if args.single_init:
        # Paranoia check: fetch ONE init's PL+SFC MARS pair, inspect the
        # resulting xarray, do not touch the on-disk zarr store.
        dt = all_inits[0]
        return _run_single_init_paranoia(dt)

    out_path = Path(args.out_zarr)
    if not args.skip_init_store:
        init_zarr_store(out_path, all_inits)

    if args.inits:
        wanted = {datetime.fromisoformat(s.strip()) for s in args.inits.split(",")}
        inits_to_do = [(i, dt) for i, dt in enumerate(all_inits) if dt in wanted]
    else:
        inits_to_do = list(enumerate(all_inits))

    progress_file = out_path.parent / f"{out_path.stem}_progress.log"
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail, n_skip = 0, 0, 0
    t0_total = datetime.now()
    n = len(inits_to_do)
    for i, (idx, dt) in enumerate(inits_to_do, start=1):
        t0 = datetime.now()
        prefix = f"[{i:>3d}/{n}] init={dt}"
        try:
            was_present = already_written(out_path, dt)
            ok = fetch_and_write_one(out_path, dt, idx)
            elapsed = datetime.now() - t0
            if was_present:
                n_skip += 1
                logging.info("%s SKIP (already written)  elapsed=%s", prefix, elapsed)
            else:
                n_ok += int(ok)
                logging.info("%s OK  elapsed=%s", prefix, elapsed)
            with progress_file.open("a") as pf:
                pf.write(f"{datetime.now().isoformat()} {prefix} "
                         f"{'SKIP' if was_present else 'OK' if ok else 'FAIL'} "
                         f"elapsed={elapsed}\n")
        except KeyboardInterrupt:
            logging.warning("%s interrupted by user", prefix)
            break
        except Exception as exc:
            n_fail += 1
            logging.exception("%s FAILED: %s", prefix, exc)
            with progress_file.open("a") as pf:
                pf.write(f"{datetime.now().isoformat()} {prefix} FAIL "
                         f"err={exc!r}\n")
            # Continue with next init -- don't let one bad init kill the run.

    total = datetime.now() - t0_total
    logging.info("Done. ok=%d skipped=%d failed=%d total_time=%s",
                 n_ok, n_skip, n_fail, total)
    return 0 if n_fail == 0 else 1


def _run_single_init_paranoia(init_dt: datetime) -> int:
    """Fetch one (date, time) PL+SFC pair and print summary. No disk write."""
    logging.info("=" * 60)
    logging.info("PARANOIA CHECK: single-init test for %s", init_dt)
    logging.info("=" * 60)

    date_token = init_dt.strftime("%Y-%m-%d")
    hour_token = init_dt.strftime("%H")
    member_token = "/".join(str(i) for i in range(1, N_MEMBERS + 1))

    common = {
        "area": "/".join(str(x) for x in AREA),
        "class": "od",
        "date": date_token,
        "expver": "1",
        "expect": "any",   # tolerate missing param/level combos
        "grid": "/".join(str(x) for x in GRID),
        "number": member_token,
        "step": "0",
        "stream": "enfo",
        "time": hour_token,
        "type": "pf",
    }

    pl_req = {**common, "levtype": "pl",
              "levelist": "/".join(str(L) for L in PL_LEVELS),
              "param": "/".join(PL_PARAMS.keys())}
    sfc_req = {**common, "levtype": "sfc",
               "param": "/".join(SFC_PARAMS.keys())}

    print()
    print(">>> MARS PL request <<<")
    for k, v in pl_req.items():
        print(f"  {k}: {v[:80] + '...' if len(str(v)) > 80 else v}")
    print()
    print(">>> MARS SFC request <<<")
    for k, v in sfc_req.items():
        print(f"  {k}: {v[:80] + '...' if len(str(v)) > 80 else v}")

    print()
    print(">>> Submitting MARS PL request (will block on tape)... <<<")
    pl_src = earthkit.data.from_source("mars", pl_req, lazily=True)
    pl_ds = pl_src.to_xarray(chunks={"number": -1, "step": 1, "level": 1})
    print("PL dataset summary:")
    print(f"  dims:   {dict(pl_ds.sizes)}")
    print(f"  vars:   {sorted(pl_ds.data_vars)}")
    print(f"  coords: {sorted(pl_ds.coords)}")
    if "level" in pl_ds.coords or "isobaricInhPa" in pl_ds.coords:
        lev_name = "isobaricInhPa" if "isobaricInhPa" in pl_ds.coords else "level"
        print(f"  levels: {sorted(pl_ds[lev_name].values.tolist())}")
    if "number" in pl_ds.dims:
        print(f"  members count: {pl_ds.sizes['number']}")

    print()
    print(">>> Submitting MARS SFC request... <<<")
    sfc_src = earthkit.data.from_source("mars", sfc_req, lazily=True)
    sfc_ds = sfc_src.to_xarray(chunks={"number": -1, "step": 1})
    print("SFC dataset summary:")
    print(f"  dims:   {dict(sfc_ds.sizes)}")
    print(f"  vars:   {sorted(sfc_ds.data_vars)}")
    print(f"  coords: {sorted(sfc_ds.coords)}")

    print()
    print(">>> Sanity: sample values for one member ===")
    # Try to take a sample value at member 0, central pixel
    try:
        m0 = pl_ds.isel(number=0) if "number" in pl_ds.dims else pl_ds
        for v in sorted(pl_ds.data_vars)[:3]:
            slab = m0[v].isel(**{k: 0 for k in m0[v].dims if k != "latitude" and k != "longitude"})
            arr = slab.values
            print(f"  PL.{v}: shape={arr.shape}, mean={np.nanmean(arr):.3g}, "
                  f"std={np.nanstd(arr):.3g}, NaN_frac={float(np.isnan(arr).mean()):.3f}")
    except Exception as exc:
        print(f"  PL sample failed: {exc}")

    try:
        m0 = sfc_ds.isel(number=0) if "number" in sfc_ds.dims else sfc_ds
        for v in sorted(sfc_ds.data_vars)[:3]:
            slab = m0[v].isel(**{k: 0 for k in m0[v].dims if k != "latitude" and k != "longitude"})
            arr = slab.values
            print(f"  SFC.{v}: shape={arr.shape}, mean={np.nanmean(arr):.3g}, "
                  f"std={np.nanstd(arr):.3g}, NaN_frac={float(np.isnan(arr).mean()):.3f}")
    except Exception as exc:
        print(f"  SFC sample failed: {exc}")

    print()
    print(">>> Paranoia check OK. No disk write performed. <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())
