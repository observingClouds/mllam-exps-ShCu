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
Write a transformed copy of a single variable of a mllam-data-prep datastore
to a replacement zarr store, to be injected with
`create_virtual_subset.py --replacement-variable/--replacement-zarr-store`.

The output mirrors the layout of the existing replacement stores (e.g.
`datastore.interior.domain03.boxcoxrain.zarr`): zarr v2, chunks of one time
step over the full grid, blosc/zstd, plus the train statistics of the
transformed variable (`mean`, `std` and one-step `diff_mean`, `diff_std`).
For interior (state) stores a stats-only store in the format of
src/compute_diff_stats.py can be written with `--stats-output` for
`create_virtual_subset.py --stats-zarr-store`.

Transforms (applied to max(x, 0)):
    log1p   log(1 + x / x0)
    mask    1 where x >= threshold, else 0
    coarse  block mean over square boxes of `--coarse-km` (local
            equirectangular projection of clon/clat), then log1p
    lag     x(t - lag_steps) (first steps repeat x(0)), then log1p

Call the script like:
    uv run --offline --script src/create_transformed_store.py \
        --input data/experiment/data/datastore.interior.domain03.zarr \
        --variable rain_gsp_rate --transform log1p --x0 1e-5 \
        --output data/experiment/data/datastore.interior.domain03.log1prain_1e-5.zarr \
        --stats-output data/experiment/data/datastore.interior.domain03.log1prain_1e-5.stats-v2.zarr
"""
import argparse
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from loguru import logger
from numcodecs import Blosc

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--input", "-i", required=True,
                    help="Path to the mllam-data-prep datastore zarr")
parser.add_argument("--output", "-o", required=True,
                    help="Path to the output replacement zarr")
parser.add_argument("--variable", "-v", required=True,
                    help="State or forcing feature to transform")
parser.add_argument("--transform", required=True,
                    choices=["log1p", "mask", "coarse", "lag"])
parser.add_argument("--x0", type=float, default=1e-5,
                    help="Scale of log1p(x / x0) (default: 1e-5)")
parser.add_argument("--threshold", type=float, default=2.8e-5,
                    help="Threshold of the mask transform (default: 2.8e-5, "
                         "i.e. 0.1 mm/h for rain_gsp_rate)")
parser.add_argument("--coarse-km", type=float, default=None,
                    help="Box size of the coarse transform in km")
parser.add_argument("--lag-steps", type=int, default=None,
                    help="Number of time steps to lag by in the lag transform")
parser.add_argument("--stats-output", default=None,
                    help="Path to an optional stats-only zarr with the one-step "
                         "diff statistics (state features only)")
parser.add_argument("--split", default="train",
                    help="Split to compute the statistics over (default: train)")
parser.add_argument("--time-chunk", type=int, default=16,
                    help="Number of time steps per dask chunk (default: 16)")
parser.add_argument("--num-workers", type=int, default=16,
                    help="Number of dask threads (default: 16)")
parser.add_argument("--max-time-steps", type=int, default=None,
                    help="Only write the first N time steps (for testing)")
args = parser.parse_args()

EARTH_RADIUS_KM = 6371.0

ds = xr.open_zarr(args.input)
if args.max_time_steps is not None:
    ds = ds.isel(time=slice(0, args.max_time_steps))
if "state" in ds and args.variable in ds.state_feature.values:
    category = "state"
elif "forcing" in ds and args.variable in ds.forcing_feature.values:
    category = "forcing"
else:
    raise ValueError(f"{args.variable} is neither a state nor a forcing feature "
                     f"of {args.input}")
feature_dim = f"{category}_feature"
da = ds[category].sel({feature_dim: [args.variable]})
da = da.chunk({feature_dim: 1, "time": args.time_chunk, "grid_index": -1})
x = np.maximum(da.astype("float64"), 0)


def log1p(v):
    return np.log1p(v / args.x0)


if args.transform == "log1p":
    y = log1p(x)
    description = f"log(1 + x / {args.x0:g})"
elif args.transform == "mask":
    y = (x >= args.threshold).astype("float64")
    description = f"1 where x >= {args.threshold:g}, else 0"
elif args.transform == "coarse":
    if args.coarse_km is None:
        raise ValueError("--coarse-km is required for the coarse transform")
    clon, clat = ds.clon.values, ds.clat.values
    x_km = EARTH_RADIUS_KM * np.cos(clat.mean()) * (clon - clon.min())
    y_km = EARTH_RADIUS_KM * (clat - clat.min())
    ix = np.floor(x_km / args.coarse_km).astype(int)
    iy = np.floor(y_km / args.coarse_km).astype(int)
    box = ix * (iy.max() + 1) + iy
    n_box = box.max() + 1
    counts = np.bincount(box, minlength=n_box)
    logger.info(f"{np.count_nonzero(counts)} boxes of {args.coarse_km:g} km, "
                f"median {np.median(counts[counts > 0]):.0f} grid points per box")

    def box_mean(block):
        out = np.empty_like(block)
        for i in np.ndindex(block.shape[:-1]):
            sums = np.bincount(box, weights=block[i], minlength=n_box)
            out[i] = (sums / np.maximum(counts, 1))[box]
        return out

    y = log1p(xr.apply_ufunc(box_mean, x, dask="parallelized",
                             output_dtypes=["float64"]))
    description = (f"log(1 + x / {args.x0:g}) of the mean over "
                   f"{args.coarse_km:g} km boxes")
elif args.transform == "lag":
    if args.lag_steps is None:
        raise ValueError("--lag-steps is required for the lag transform")
    lagged_index = np.maximum(np.arange(ds.sizes["time"]) - args.lag_steps, 0)
    x_lagged = x.isel(time=lagged_index).assign_coords(time=x.time)
    y = log1p(x_lagged.chunk({"time": args.time_chunk}))
    description = (f"log(1 + x / {args.x0:g}) of x(t - {args.lag_steps} steps), "
                   "first steps repeat x(0)")

y = y.astype(ds[category].dtype).drop_vars(["clat", "clon"], errors="ignore")
y.attrs = {**da.attrs, "transform": description,
           "transform_source": str(Path(args.input).resolve())}
y.name = category

# Same variables as the existing replacement stores: the feature coordinates,
# time, grid_index and the small coordinates of the other categories
coords = [c for c in ds.coords
          if c not in ("clat", "clon") and ds[c].ndim <= 1
          and set(ds[c].dims) <= {feature_dim, "time", "grid_index", "split_name",
                                  "split_part", "forcing_feature", "static_feature"}]
if category == "forcing":
    # Boundary replacement stores carry no static feature labels: the virtual
    # concatenation would have to compare them with those of the input
    coords = [c for c in coords if not c.startswith("static_feature_")]
ds_out = xr.Dataset({category: y}).assign_coords(
    {c: ds[c].sel({feature_dim: [args.variable]}).variable
     if feature_dim in ds[c].dims else ds[c].variable for c in coords})
# Coordinates without a data variable on their dimension are only kept as
# coordinates when listed in the global attribute (as mllam-data-prep does)
ds_out.attrs["coordinates"] = " ".join(
    c for c in ds_out.coords
    if c not in ds_out.dims and not set(ds_out[c].dims) & set(y.dims))
for var in ds_out.variables.values():
    var.encoding = {}
compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
encoding = {category: {"chunks": (1, 1, ds.sizes["grid_index"]),
                       "compressors": [compressor]}}

dask.config.set(scheduler="threads", num_workers=args.num_workers)
logger.info(f"Writing {args.variable} ({category}) as {description} to {args.output}")
ds_out.to_zarr(args.output, mode="w", zarr_format=2, consolidated=True,
               encoding=encoding)

# Statistics from the written store, in float64 over the split
split_start, split_end = ds.splits.sel(split_name=args.split).values
written = xr.open_zarr(args.output)[category].sel(time=slice(split_start, split_end))
written = written.astype("float64").chunk({"time": args.time_chunk})
reduce_dims = ["time", "grid_index"]
diff = written.diff(dim="time")
mean, std, diff_mean, diff_std = dask.compute(
    written.mean(dim=reduce_dims), written.std(dim=reduce_dims),
    diff.mean(dim=reduce_dims), diff.std(dim=reduce_dims))
prefix = f"{category}__{args.split}"
stats = xr.Dataset({f"{prefix}__{op}": values.astype(ds[f"{prefix}__{op}"].dtype).drop_vars(
    [c for c in values.coords if c != feature_dim])
    for op, values in [("mean", mean), ("std", std),
                       ("diff_mean", diff_mean), ("diff_std", diff_std)]})
for var in stats.variables.values():
    var.encoding = {}
# Appending replaces the global attributes
stats.attrs = ds_out.attrs
# Same codec as the statistics of the input, as the virtual concatenation in
# create_virtual_subset.py requires identical codecs
stats.drop_vars(feature_dim).to_zarr(
    args.output, mode="a", zarr_format=2, consolidated=True,
    encoding={v: {"compressors": [compressor]} for v in stats.data_vars})
logger.info(f"{args.variable}: " + "  ".join(
    f"{op} {float(stats[f'{prefix}__{op}'].values[0]):.6g}"
    for op in ["mean", "std", "diff_mean", "diff_std"]))

if args.stats_output is not None:
    if category != "state":
        raise ValueError("--stats-output is only supported for state features")
    ds_stats = stats[[f"{prefix}__diff_mean", f"{prefix}__diff_std"]].assign_coords(
        {c: ds[c].sel(state_feature=[args.variable]).load()
         for c in ds[f"{prefix}__diff_std"].coords})
    ds_stats.attrs = {
        "source_dataset": str(Path(args.output).resolve()),
        "split": args.split,
        "split_start": str(split_start),
        "split_end": str(split_end),
        "n_time_steps": int(written.sizes["time"]),
        "description": ("One-step time-difference statistics of the transformed "
                        "variable, computed with src/create_transformed_store.py"),
    }
    for var in ds_stats.variables.values():
        var.encoding = {}
    ds_stats.to_zarr(args.stats_output, mode="w", consolidated=True)
    logger.info(f"Written {args.stats_output}")
