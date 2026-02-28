#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/analysis/.venv/bin/python
"""Create percentile summaries for predictions and truth.

The script reads one or more prediction zarr datasets along with the
ground-truth zarr, computes percentiles of the ``t_2m`` field over the
spatial grid and returns two xarray datasets:

* **predictions** with dims ``source``, ``variable``, ``percentile``
  plus the original ``start_time`` and ``elapsed_forecast_duration``
* **ground truth** with dims ``source``, ``variable``, ``percentile``
  and ``time``

Only the ``t_2m`` variable is handled at the moment.

Example usage::

    python compute_percentiles.py \
        --predictions Baseline=../evals/baseline_unroll30.zarr \
                      "Bare-minimum=../evals/bare-minimum_unroll30.zarr" \
        --ground_truth ../data/experiment/data/datastore.interior.domain03.zarr

The outputs are written by default to ``pred_percentiles.nc`` and
``gt_percentiles.nc``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# bring in dask/distributed for parallel compute
from dask.distributed import Client, LocalCluster

# helper lives in ../src/helpers
sys.path.append("../src/helpers")
from analysis_helpers import combine_state_features

from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from shapely import vectorized

def get_interior_domain_mask(ds, margin):
    """Return cell index mask for interior domain with given margin."""
    points = np.column_stack([ds.clon.values.ravel(), ds.clat.values.ravel()])
    hull = MultiPoint(points).convex_hull
    inner_polygon = hull.buffer(-margin)
    mask = vectorized.contains(inner_polygon, ds.clon.values, ds.clat.values)
    return mask


def compute_prediction_percentiles(prediction_paths, percentiles, mask=None):
    """Return a dataset of percentiles for every source.

    Parameters
    ----------
    prediction_paths : dict
        Mapping from source name to zarr path.
    percentiles : array_like
        Quantile levels between 0 and 1.

    Returns
    -------
    xr.Dataset
        DataArray ``t_2m`` with dimensions
        ``source, variable, percentile, start_time, elapsed_forecast_duration``.
    """
    components = []
    for name, path in prediction_paths.items():
        print(f"loading prediction {name}: {path}")
        ds = combine_state_features(
            xr.open_dataset(path, engine="zarr", chunks="auto")
        )
        # only keep t_2m
        var = ds["t_2m"]
        if mask is not None:
            var = var.isel(grid_index=mask)
        # compute quantiles along the spatial dimension
        q = var.quantile(percentiles, dim="grid_index")
        # rename/xarray conventions
        q = q.rename({"quantile": "percentile"})
        q = q.expand_dims({"source": [name], "variable": ["t_2m"]})
        components.append(q)

    return xr.concat(components, dim="source")


def compute_ground_truth_percentiles(gt_path, percentiles, mask=None):
    """Return percentiles for ground truth dataset.

    The resulting dataset has dimensions ``source, variable, percentile, time``
    and a singleton ``source`` coordinate of ``"ground_truth"``.
    """
    print(f"loading ground truth: {gt_path}")
    ds = combine_state_features(
        xr.open_dataset(gt_path, engine="zarr", chunks="auto")
    )
    if mask is not None:
        ds = ds.isel(grid_index=mask)
    var = ds["t_2m"]
    q = var.quantile(percentiles, dim="grid_index")
    q = q.rename({"quantile": "percentile"})
    q = q.expand_dims({"source": ["ground_truth"], "variable": ["t_2m"]})
    return q


def main():
    parser = argparse.ArgumentParser(
        description="Compute percentile summaries for prediction and truth datasets."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=False,
        help="Source=name:path pairs for prediction zarr datasets",
    )
    parser.add_argument(
        "--ground_truth",
        required=False,
        help="Path to ground truth zarr dataset",
    )
    parser.add_argument(
        "--output",
        default="percentiles.nc",
        help="Output netCDF file for percentiles",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0,
        help="Margin in degrees to exclude from edges (default: 0.005)",
        required=False
    )

    args = parser.parse_args()

    cluster = LocalCluster()
    client = Client(cluster)
    print("Dask dashboard available at:", client.dashboard_link)

    percentiles = np.linspace(0, 1, 101)

    if args.margin > 0:
        margin = args.margin
        ds = xr.open_dataset("../data/experiment/data/datastore.interior.domain03.zarr", engine="zarr")
        mask = get_interior_domain_mask(ds, margin)

    # parse predictions argument
    prediction_paths = {}
    if args.predictions:
        for item in args.predictions:
            if "=" not in item:
                raise ValueError(f"invalid prediction spec '{item}', expected name=path")
            name, path = item.split("=", 1)
            prediction_paths[name] = path
        pred_ds = compute_prediction_percentiles(prediction_paths, percentiles, mask=mask if args.margin > 0 else None)
        print(f"computing and writing percentiles to {args.output}")
        pred_ds.to_netcdf(args.output)

    if args.ground_truth:
        gt_ds = compute_ground_truth_percentiles(args.ground_truth, percentiles, mask=mask if args.margin > 0 else None)
        print(f"computing and writing percentiles to {args.output}")
        gt_ds.to_netcdf(args.output)

    print("shutting down dask client and cluster")
    client.close()
    cluster.close()

    print("done")


if __name__ == "__main__":
    main()
