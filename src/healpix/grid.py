
from intake import open_catalog
import xarray as xr
import uxarray as ux


cat = open_catalog("https://raw.githubusercontent.com/observingClouds/eurec4a-intake/refs/heads/add/ICON-LES_DOM02_synsat_native/catalog.yml")

data = xr.open_dataset("test.zarr", engine="zarr")
grid = xr.open_dataset("grid.nc")  #cat.simulations.grids[data.uuidOfHGrid].to_dask()

ugrid = ux.open_grid(grid)

print("Creating uxarray")
ds = xr.merge([data, grid])
uxds = ux.UxDataset(data, uxgrid=ugrid)

print("Creating healpix grid")
hp_grid = ux.Grid.from_healpix(zoom=8)  # zoom level of 12 corresponds to 

print("Converting to healpix")
uxds_hp = uxds.remap.nearest_neighbor(hp_grid)

print("Writing to zarr")
uxds_hp.to_zarr("test_hp.zarr", mode="w")