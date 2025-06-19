"""Add additional grid information to dataset.

The original dataset created in preprocess.py has no grid information.
This script adds cartesian position and also latitude and longitude
information to the dataset.
"""
import xarray as xr
import numpy as np
from intake import open_catalog

cat = open_catalog("https://raw.githubusercontent.com/observingClouds/eurec4a-intake/refs/heads/add/ICON-LES_DOM02_synsat_native/catalog.yml")
data = cat.simulations.ICON.LES_CampaignDomain_control.surface_DOM02.to_dask()
grid = cat.simulations.grids[data.uuidOfHGrid].to_dask()

R = 6371e3

x = R * np.cos(grid.clat) * np.cos(grid.clon)
y = R * np.cos(grid.clat) * np.sin(grid.clon)
z = R * np.sin(grid.clat)

ds_grid = xr.Dataset({"x":x, "y":y, "z":z}, coords={"cell":grid.cell})
ds_grid = ds_grid.set_coords(["x", "y", "z"])
ds_grid.to_zarr("data/grid.zarr", mode="w")
print("Please move the folders within grid.zarr into the zarr directory requiring the grid info. Add `\"coordinates\": \"clat clon x y z\"` to the .zattrs")
