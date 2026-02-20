#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/.venv/bin/python
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=30G
#SBATCH --gres=tmpfs:100G
#SBATCH -t 4:00:00
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import yaml
import cartopy.crs as ccrs

import sys
sys.path.append("../src")
import energy_spectra as es
import copy
import argparse

def projection(projection_info):
    """Create a cartopy projection from a projection info dictionary.
    """
    class_name = projection_info["class_name"]
    ProjectionClass = getattr(ccrs, class_name)
    # need to copy otherwise we modify the dict stored in the dataclass
    # in-place
    kwargs = copy.deepcopy(projection_info["kwargs"])

    globe_kwargs = kwargs.pop("globe", {})
    if len(globe_kwargs) > 0:
        kwargs["globe"] = ccrs.Globe(**globe_kwargs)
    
    return ProjectionClass(**kwargs)

def combine_state_features(ds):
    """
    Combine individual state features from an xarray.Dataset into a single dataset.

    Parameters:
    ds (xarray.Dataset): The input dataset containing state features.

    Returns:
    xarray.Dataset: A new dataset with each state feature as a separate variable.
    """
    # Create a new dataset to hold the combined features
    combined_ds = xr.Dataset()

    # Loop through each state feature and add it to the combined dataset
    for feature in ds.state_feature.values:
        individual_array = ds.state.sel(state_feature=feature)
        combined_ds[feature] = individual_array

    return combined_ds

if __name__ == "__main__":
    # command-line arguments
    parser = argparse.ArgumentParser(description="Compute energy spectra")
    parser.add_argument("--prediction", dest="path_mllam",
                        default="../evals/rain_mse_boxcox_unroll100.zarr",
                        help="Path to the mllam evaluation zarr dataset")
    args = parser.parse_args()

    print("Loading datasets...", flush=True)
    ds_interior = xr.open_dataset("../data/experiment/data/datastore.interior.domain03.zarr", engine="zarr")
    creation_config_yaml = yaml.safe_load(ds_interior.attrs["creation_config"])
    del ds_interior.attrs["creation_config"]
    lats = np.rad2deg(ds_interior["clat"])
    lons = np.rad2deg(ds_interior["clon"])
    ds_interior = combine_state_features(ds_interior)

    path_mllam = args.path_mllam
    ds_mllam = xr.open_zarr(path_mllam, consolidated=False)
    ds_mllam = combine_state_features(ds_mllam)

    ds_mllam = ds_mllam.isel(elapsed_forecast_duration=[0, 1, 2, 3, 4, 5, 12, 29])
    ds_interior = ds_interior.sel(time=slice(ds_mllam.start_time.min(), ds_mllam.start_time.max() + np.timedelta64(29, "h")))

    print("Defining projection and extracting grid coordinates...", flush=True)
    model_proj = projection(creation_config_yaml["extra"]["projection"])

    # Extract unstructured grid coordinates
    # Adjust these variable names if needed
    grid_lat = np.rad2deg(ds_interior['clat'].isel(grid_index=slice(0,-1)).values)
    grid_lon = np.rad2deg(ds_interior['clon'].isel(grid_index=slice(0,-1)).values)

    xyz = model_proj.transform_points(
            ccrs.PlateCarree(), grid_lon, grid_lat
        )

    # projected X, Y
    # Stack coordinates for KDTree
    grid_points = np.column_stack((xyz[:,0], xyz[:,1]))
    center_pos = grid_points.mean(axis=0)

    x = np.arange(center_pos[0]-60000, center_pos[0]+60000, 600)
    y = np.arange(center_pos[1]-98000, center_pos[1]+98000, 600)
    target_x, target_y = np.meshgrid(x, y)
    target_points = np.column_stack((target_x.flatten(), target_y.flatten()))
    # Build KDTree and query nearest neighbors
    max_dist = 600  # example: meters, adjust as needed
    tree = cKDTree(grid_points)
    distances, idx = tree.query(target_points)

    print("Regridding data onto regular grid...", flush=True)
    gt_regridded_data = {}
    variables = ['t_2m',] # 'qv_2m', 'u_10m', 'v_10m', 'tqc_dia', 'pres_sfc'
    for var in variables:
        data = ds_interior[var].isel(grid_index=idx).data.reshape(len(ds_interior.time), len(y), len(x))
        # Map data from unstructured grid to regular grid
        # regridded[distances.reshape(len(y), len(x), -1) > max_dist] = np.nan  # Mask out points beyond max_dist
        gt_regridded_data[var] = (["time", "y", "x"], data)

    gt_regridded = xr.Dataset(
        gt_regridded_data,
        coords={"y": y, "x": x, "time": ds_interior.time.values}
    )

    print("Ground truth regridded", flush=True)

    mllam_regridded_data = {}
    for var in variables:
        data = ds_mllam[var].isel(grid_index=idx).data.reshape(len(ds_mllam.start_time), len(ds_mllam.elapsed_forecast_duration), len(y), len(x))
        # Map data from unstructured grid to regular grid
        # regridded[distances.reshape(len(y), len(x), -1) > max_dist] = np.nan  # Mask out points beyond max_dist
        mllam_regridded_data[var] = (["start_time", "elapsed_forecast_duration", "y", "x"], data)
    mllam_regridded = xr.Dataset(
        mllam_regridded_data,
        coords={"y": y, "x": x, "start": ds_mllam.start_time.values, "elapsed_forecast_duration": ds_mllam.elapsed_forecast_duration.values}
    )

    spectra_cache = es.calculate_all_spectra(
        gt_regridded, mllam_regridded, None, variables=["t_2m",] # "u_10m", "v_10m", "qv_2m", "tqc_dia", "pres_sfc"]
    )

    print("Writing spectra cache to disk...")

    # Save as compressed numpy archive (keeps nested dict structure and
    # matches how the notebooks load `spectra_cache.npz` with allow_pickle=True)
    out_base = args.path_mllam.split('/')[-1].split('.')[0]
    out_npz = f"spectra_cache_{out_base}.npz"
    np.savez_compressed(out_npz, **spectra_cache)
    print(f"Saved {out_npz}")

