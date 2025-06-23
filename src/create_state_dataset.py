"""Helper script to create a static dataset for neural-lam which is currently required for training.

The seasurface temperature is used to derive a land-sea mask.
In order to get a standard divitation different to 0 that causes issues (https://github.com/mllam/neural-lam/issues/136),
a small noise is added to the land-sea mask.
"""
import xarray as xr
import numpy as np
import argparse
from intake import open_catalog

parser = argparse.ArgumentParser(description="Create a static dataset for neural-lam.")
parser.add_argument("--input", type=str, help="Path to the input Zarr dataset.", default="../data/test.zarr")
parser.add_argument("--output", type=str, help="Path to the output Zarr dataset.", default="../data/lsm_DOM02.zarr")
args = parser.parse_args()

cat = open_catalog("https://raw.githubusercontent.com/observingClouds/eurec4a-intake/refs/heads/remove/compression/catalog.yml")

ds = cat.simulations.ICON.LES_CampaignDomain_control.surface_DOM02.to_dask()


#ds = xr.open_zarr(args.input)
r=np.random.randn(len(ds.cell))/1e3


ds_out = ds[['cell']]
ds_out['lsm'] = (xr.where(ds['t_seasfc'].isel(time=0).isnull(), 0., 1.)*r).rename("lsm")
ds_out.drop_vars("time").squeeze(drop=True).to_zarr(args.output, mode="w", consolidated=True)
