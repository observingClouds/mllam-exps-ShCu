"""Helper script to create a static dataset for neural-lam which is currently required for training.

The seasurface temperature is used to derive a land-sea mask.
In order to get a standard divitation different to 0 that causes issues (https://github.com/mllam/neural-lam/issues/136),
a small noise is added to the land-sea mask.
"""
import xarray as xr
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Create a static dataset for neural-lam.")
parser.add_argument("input_path", type=str, help="Path to the input Zarr dataset.", default="../data/test.zarr")
parser.add_argument("output_path", type=str, help="Path to the output Zarr dataset.", default="../data/lsm_DOM02.zarr")
args = parser.parse_args()

ds = xr.open_zarr(args.input_path)
r=np.random.randn(len(ds.cell))/1e3

(xr.where(ds.t_seasfc.isel(time=0).isnull(), 0., 1.)*r).rename("lsm").drop_vars("time").squeeze(drop=True).to_zarr(args.output_path, mode="w", consolidated=True)