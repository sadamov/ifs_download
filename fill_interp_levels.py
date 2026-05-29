"""Post-download log-pressure interpolation for the 2 PL levels not archived
by MARS for type=pf step=0 (150 + 600 hPa).

Run AFTER download_ic_perturbed.py finishes. Opens the IC perturbed zarr,
reads adjacent archived levels per (init_time, ensemble) slab, computes
f(150) = 0.42*f(100) + 0.58*f(200) and f(600) = 0.46*f(500) + 0.54*f(700)
in log-pressure linear, writes back via region.

Idempotent: skips levels already non-NaN.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PL_PARAMS = ("t", "u", "v", "q", "z", "w")

# Each entry: target_level: (lower_archived, upper_archived, alpha)
# alpha = (log(target) - log(lower)) / (log(upper) - log(lower))
INTERP = {
    150: (100, 200, (np.log(150) - np.log(100)) / (np.log(200) - np.log(100))),
    600: (500, 700, (np.log(600) - np.log(500)) / (np.log(700) - np.log(500))),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", default="/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr")
    p.add_argument("--init-slice", default=None,
                   help="Optional init_time slice like '0:10' for testing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = Path(args.zarr)

    ds = xr.open_zarr(out, consolidated=False)
    n_init = ds.sizes["init_time"]
    init_range = range(n_init)
    if args.init_slice:
        a, b = args.init_slice.split(":")
        init_range = range(int(a) if a else 0, int(b) if b else n_init)

    for new_L, (lo, hi, alpha) in INTERP.items():
        li_new = int(np.where(ds.level.values == new_L)[0][0])
        li_lo = int(np.where(ds.level.values == lo)[0][0])
        li_hi = int(np.where(ds.level.values == hi)[0][0])
        logging.info("interp L=%d using L=%d, L=%d, alpha=%.4f (positions %d <- %d, %d)",
                     new_L, lo, hi, alpha, li_new, li_lo, li_hi)

        for i in init_range:
            done = []
            for v in PL_PARAMS:
                arr_new = ds[v].isel(init_time=i, level=li_new, ensemble=0).values
                if np.all(np.isfinite(arr_new)) and np.any(arr_new):
                    continue
                lo_slab = ds[v].isel(init_time=i, level=li_lo).load()
                hi_slab = ds[v].isel(init_time=i, level=li_hi).load()
                interp = ((1 - alpha) * lo_slab + alpha * hi_slab).astype("float32")
                interp = interp.expand_dims({"init_time": [ds.init_time.values[i]],
                                              "level": [new_L]})
                interp = interp.drop_vars([c for c in list(interp.coords)
                                            if "init_time" not in interp[c].dims])
                interp.to_dataset(name=v).to_zarr(
                    out,
                    region={"init_time": slice(i, i + 1), "level": slice(li_new, li_new + 1)},
                    consolidated=False,
                )
                done.append(v)
            if done:
                logging.info("  init %d (%s): filled %s", i, str(ds.init_time.values[i])[:13], done)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
