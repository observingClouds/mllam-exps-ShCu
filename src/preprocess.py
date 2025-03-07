from intake import open_catalog
from loguru import logger
import xarray as xr
import pandas as pd
import numpy as np
cat = open_catalog("https://raw.githubusercontent.com/observingClouds/eurec4a-intake/refs/heads/add/ICON-LES_DOM02_synsat_native/catalog.yml")

ds_surface = cat.simulations.ICON.LES_CampaignDomain_control.surface_DOM02.to_dask()
ds_rttov = cat.simulations.ICON.LES_CampaignDomain_control.rttov_DOM02_native.to_dask()
ds_radiation = cat.simulations.ICON.LES_CampaignDomain_control.radiation_DOM02.to_dask()

times_expected = pd.date_range("2020-01-11T22:20:00", "2020-02-19T10:00:00", freq='10min')
times_sfc = pd.to_datetime(ds_surface.time.values)
times_rttov = pd.to_datetime(ds_rttov.time.values)
times_radiation = pd.to_datetime(ds_radiation.time.values)
common_times = set(times_expected).intersection(times_sfc).intersection(times_rttov).intersection(times_radiation)

missing_times = set(times_expected).difference(common_times)
assert len(missing_times) == 0, f"Missing times: {missing_times}"
assert len(common_times) == len(times_expected), f"Missing times: {set(times_expected).difference(common_times)}"

logger.info(f"{len(common_times)} times are common to all datasets, starting on {times_expected[0]} and ending on {times_expected[-1]}")

# Simplify time selection
time_slice = slice(min(common_times), max(common_times))
times_list = np.array(sorted(list(common_times)))

# Subset datasets
ds_surface = ds_surface.sel(time=times_list)
ds_rttov = ds_rttov.sel(time=times_list)
ds_radiation = ds_radiation.sel(time=times_list)
assert ds_surface.time.equals(ds_rttov.time), "Time coordinates are not equal"
assert ds_surface.time.equals(ds_radiation.time), "Time coordinates are not equal"

merged_ds = xr.merge([ds_surface, ds_rttov, ds_radiation])

# Select variables
variables = ["clct", "lhfl_s", "pres_sfc", "qv_2m", "rain_gsp_rate", "shfl_s", "t_2m", 
             "t_seasfc", "tot_prec", "tot_prec", "tqc_dia", "tqi_dia", "tqv_dia", "u_10m",
             "v_10m", "sob_t", "sod_t", "sou_t", "thb_t", "synsat_rttov_forward_model_2__abi_ir__goes_16__channel_7",
]
ds = merged_ds[variables]
logger.info("Rechunking")
ds = ds.isel(time=slice(0,10)).chunk({'time':1, 'cell':-1})

for var in ds.data_vars:
    del ds[var].encoding["chunks"]
    del ds[var].encoding["preferred_chunks"]
for var in ds.variables:
    try:
        del ds[var].encoding["compressors"]
    except KeyError:
        continue

logger.info("Writing to Zarr")
ds.isel(time=slice(0, 10)).to_zarr("test.zarr", mode="w")

# # Splitting domain into subdomains
# ds_grid = cat.simulations.grids[ds.attrs["uuidOfHGrid"]].to_dask()



# import os
# import pickle
# import types

# import numpy as np


# def grid_selection(
#     grid,
#     lats=types.MappingProxyType({"lats": [11, 15]}),
#     lons=types.MappingProxyType({"lons": [-59.3, -55.3]}),
# ):
#     # Subsection
#     x_range = lons
#     y_range = lats

#     # Create grid-mask
#     cell = (
#         (grid.clat.values >= np.deg2rad(y_range[0]))
#         & (grid.clat.values <= np.deg2rad(y_range[1]))
#         & (grid.clon.values >= np.deg2rad(x_range[0]))
#         & (grid.clon.values <= np.deg2rad(x_range[1]))
#     )
#     # vert = (
#     #     (grid.vlon.values >= np.deg2rad(x_range[0]))
#     #     & (grid.vlon.values <= np.deg2rad(x_range[1]))
#     #     & (grid.vlat.values >= np.deg2rad(y_range[0]))
#     #     & (grid.vlat.values <= np.deg2rad(y_range[1]))
#     # )
#     # cell = vert
#     grid = grid.sel(vertex=cell)
#     return cell, grid


# def load_grid_subset(dom, lats, lons, grid, path=".", return_grid=False):
#     pkl_filename = os.path.join(
#         path,
#         f"cells_DOM0{dom}_lats{'-'.join(map(str,lats))}_lons{'-'.join(map(str,lons))}.pkl",
#     )
#     if not os.path.exists(pkl_filename) or return_grid:
#         print("Creating cell mask")
#         cell_subsection, grid_subsection = grid_selection(grid, lats, lons)
#         with open(pkl_filename, "wb") as f:
#             pickle.dump(cell_subsection, f)
#     else:
#         print("Reading cell mask")
#         with open(pkl_filename, "rb") as f:
#             cell_subsection = pickle.load(f)
#     if return_grid:
#         return cell_subsection, grid_subsection
#     else:
#         return cell_subsection
    
# # _, g = load_grid_subset(2, [10,12],[-59,-57], ds_grid, return_grid=True)




# # import eurec4a
# # import xarray as xr
# # import numpy as np
# # import pandas as pd

# import matplotlib.pylab as plt
# # import datashader
# # from datashader.mpl_ext import dsshow

# import cartopy.crs as ccrs
# import cartopy.feature as cf



# # Lazy loading of output and grid
# data = cat.simulations.ICON.LES_CampaignDomain_control.surface_DOM01.to_dask()
# grid = cat.simulations.grids[data.uuidOfHGrid].to_dask()

# from uxarray.core.dataset import UxDataset
# ds = xr.merge([data, grid])
# uxds = UxDataset.from_xarray(ds)
# uxds.isel(time=100).rh_2m.plot()

# ux_ds = uxr.open_dataset(grid, data)

# central_longitude = -53.54884554550185
# central_latitude = 12.28815437976341
# satellite_height = 8225469.943160511

# # da = data[variable].sel(time='2020-02-08 12:00:00')

# projection = ccrs.NearsidePerspective(central_longitude=central_longitude, central_latitude=central_latitude, satellite_height=satellite_height)

# coords = projection.transform_points(
#     ccrs.Geodetic(),
#     np.rad2deg(grid.clon.values),
#     np.rad2deg(grid.clat.values),
# )

# fig, ax = plt.subplots(subplot_kw={"projection": projection})
# fig.canvas.draw_idle()
# ax.add_feature(cf.COASTLINE, linewidth=0.8)

# artist = ax.pcolormesh(ds_grid.isel(cell=slice(496709, 496709+4**9)).clat, ds_grid.sel(cell=slice(496709, 496709+4**9)).clon, ds.isel(time=100,cell=slice(496709, 496709+4**9)).t_2m)
