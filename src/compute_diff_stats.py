# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray",
#   "dask",
#   "zarr==3.1.6",
#   "numpy",
#   "loguru",
# ]
# ///
"""
Recompute the time-difference statistics of the state variables of a
mllam-data-prep datastore and write them to a small stats-only zarr store.

mllam-data-prep <= v0.6.1 computes `diff_std` from the second-order difference
when `diff_mean` precedes it in the list of statistics ops
(https://github.com/mllam/mllam-data-prep/issues/102). This script computes
`state__<split>__diff_mean` and `state__<split>__diff_std` from the one-step
difference, without having to recreate the datastore. The resulting store can
be referenced with `create_virtual_subset.py --stats-zarr-store`.

As a sanity check, `state__<split>__mean` and `state__<split>__std` are
recomputed as well and compared against the values stored in the input.

Call the script like:
    uv run --script src/compute_diff_stats.py \
        --input data/experiment/data/datastore.interior.domain03.zarr \
        --output data/experiment/data/datastore.interior.domain03.stats-v2.zarr
"""
import argparse
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from loguru import logger

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--input", "-i", required=True,
                    help="Path to the mllam-data-prep datastore zarr")
parser.add_argument("--output", "-o", required=True,
                    help="Path to the output stats zarr")
parser.add_argument("--split", default="train",
                    help="Split to compute the statistics over (default: train)")
parser.add_argument("--splits-from", default=None,
                    help="Datastore zarr to read the split definition from, for "
                         "stores without `splits` (default: the input)")
parser.add_argument("--time-chunk", type=int, default=16,
                    help="Number of time steps per dask chunk (default: 16)")
parser.add_argument("--max-time-steps", type=int, default=None,
                    help="Only use the first N time steps of the split (for testing)")
parser.add_argument("--num-workers", type=int, default=4,
                    help="Number of dask threads (default: 4)")
parser.add_argument("--rtol", type=float, default=1e-3,
                    help="Relative tolerance for the mean/std sanity check")
args = parser.parse_args()

ds = xr.open_zarr(args.input)
ds_splits = ds if args.splits_from is None else xr.open_zarr(args.splits_from)
split_start, split_end = ds_splits.splits.sel(split_name=args.split).values
state = ds.state.sel(time=slice(split_start, split_end))
if args.max_time_steps is not None:
    state = state.isel(time=slice(0, args.max_time_steps))

time_steps = np.unique(np.diff(state.time.values))
if len(time_steps) != 1:
    raise ValueError(
        f"Time steps in split '{args.split}' are not equidistant ({time_steps}), "
        "differences across gaps would enter the statistics"
    )
logger.info(
    f"Computing statistics over {state.sizes['time']} time steps "
    f"({state.time.values[0]} to {state.time.values[-1]}) and "
    f"{state.sizes['grid_index']} grid points"
)

dask.config.set(scheduler="threads", num_workers=args.num_workers)

# One feature at a time to bound memory; accumulate in float64 since the
# reductions run over ~1e9 values per feature
reduce_dims = ["time", "grid_index"]
results = {"diff_mean": [], "diff_std": [], "mean": [], "std": []}
for feature in state.state_feature.values:
    x = (
        state.sel(state_feature=[feature])
        .astype("float64")
        .chunk({"time": args.time_chunk, "grid_index": -1})
    )
    dx = x.diff(dim="time")
    computed_feature = dask.compute(
        dx.mean(dim=reduce_dims),
        dx.std(dim=reduce_dims),
        x.mean(dim=reduce_dims),
        x.std(dim=reduce_dims),
    )
    for op, values in zip(results, computed_feature):
        results[op].append(values)
    logger.info(f"Done with {feature}")
diff_mean, diff_std, mean, std = (
    xr.concat(results[op], dim="state_feature") for op in results
)

prefix = f"state__{args.split}"
stored = {
    "mean": ds[f"{prefix}__mean"].values,
    "std": ds[f"{prefix}__std"].values,
    "diff_mean": ds[f"{prefix}__diff_mean"].values,
    "diff_std": ds[f"{prefix}__diff_std"].values,
}
computed = {
    "mean": mean.values,
    "std": std.values,
    "diff_mean": diff_mean.values,
    "diff_std": diff_std.values,
}
for i, feature in enumerate(state.state_feature.values):
    logger.info(
        f"{feature:>14s}: "
        + "  ".join(
            f"{op} {stored[op][i]:.6g} -> {computed[op][i]:.6g}" for op in stored
        )
    )

if args.max_time_steps is None:
    mismatch = [
        f"{feature} {op}"
        for op in ["mean", "std"]
        for i, feature in enumerate(state.state_feature.values)
        if not np.isclose(stored[op][i], computed[op][i], rtol=args.rtol, atol=0)
    ]
    if mismatch:
        raise ValueError(
            f"Recomputed statistics do not match the stored ones for: {mismatch}. "
            "The split or the data might differ from what the datastore was "
            "created with."
        )

stats_vars = {}
for op, values in [("diff_mean", diff_mean), ("diff_std", diff_std)]:
    name = f"{prefix}__{op}"
    da = values.astype(ds[name].dtype)
    da.attrs = ds[name].attrs
    stats_vars[name] = da
ds_stats = xr.Dataset(stats_vars).drop_vars(
    [c for c in diff_mean.coords if c != "state_feature"]
)
ds_stats = ds_stats.assign_coords(
    {
        c: ds[c].load()
        for c in ds[f"{prefix}__diff_std"].coords
        if c != "state_feature"
    }
)
ds_stats.attrs = {
    "source_dataset": str(Path(args.input).resolve()),
    "split": args.split,
    "split_start": str(split_start),
    "split_end": str(split_end),
    "n_time_steps": int(state.sizes["time"]),
    "description": (
        "One-step time-difference statistics of the state variables, "
        "recomputed with src/compute_diff_stats.py (see "
        "https://github.com/mllam/mllam-data-prep/issues/102)"
    ),
}
if args.max_time_steps is not None:
    ds_stats.attrs["max_time_steps"] = args.max_time_steps
# Encodings inherited from the source store (e.g. zarr v2 codecs) are not
# needed for these tiny arrays and can be incompatible with the writer
for var in ds_stats.variables.values():
    var.encoding = {}
ds_stats.to_zarr(args.output, mode="w", consolidated=True)
logger.info(f"Written {args.output}")
