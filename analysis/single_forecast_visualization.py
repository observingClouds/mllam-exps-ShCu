#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/.venv/bin/python
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=10G
#SBATCH -t 0:10:00
import xarray as xr
import sys
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pandas as pd

sys.path.append("../src/helpers")
from analysis_helpers import combine_state_features

# Define prediction datasets to compare
prediction_datasets = {
    "Baseline": "../evals/rain_mse_boxcox2.zarr",
    # "Baseline + rr": "../evals/rain_mse_boxcox_unroll100.zarr",
    # "Baseline + q": "../evals/rain_mse_boxcox2.zarr",
    # "Baseline + fluxes": "../evals/rain_mse_boxcox2.zarr",
    # "Baseline + BT": "../evals/rain_mse_boxcox2.zarr",
    "Surface only": "../evals/rain_mse_boxcox2.zarr",
    "Bare-minimum": "../evals/rain_mse_boxcox2.zarr",
}

# Load predictions
predictions = {
    name: combine_state_features(xr.open_dataset(path, engine="zarr"))
    for name, path in prediction_datasets.items()
}

# Load ground truth
ground_truth = xr.open_dataset("../data/experiment/data/datastore.interior.domain03.zarr", engine="zarr")
combined_ground_truth = combine_state_features(ground_truth)

# Extract prediction data arrays for each model
da_predictions = {
    name: pred.t_2m.sel(start_time="2020-02-12T15:00:00").isel(elapsed_forecast_duration=slice(0,4))
    for name, pred in predictions.items()
}

# Use the first prediction to determine times
first_prediction = list(da_predictions.values())[0]
prediction_times = first_prediction.start_time.values + first_prediction.elapsed_forecast_duration.values
da_target = combined_ground_truth.t_2m.sel(time=prediction_times)

assert da_target.time.shape == first_prediction.elapsed_forecast_duration.shape

# Calculate global min/max across all predictions and target
vmin = min(da_target.min(), min(pred.min() for pred in da_predictions.values()))
vmax = max(da_target.max(), max(pred.max() for pred in da_predictions.values()))


# Extract the temporal dimension
time_steps = da_target.time.values
num_models = len(da_predictions)

# Create subplots for each time step: 1 ground truth + N predictions
fig, axes = plt.subplots(
    len(time_steps),
    1 + num_models,
    figsize=(6 * (1 + num_models), 4 * len(time_steps)),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# Ensure axes is a 2D array for consistent indexing
if len(time_steps) == 1:
    axes = np.array([axes])

# Plot for each time step
for i, time_step in enumerate(time_steps):
    da_target_time = da_target.isel(time=i)
    
    # Plot ground truth in first column
    ax = axes[i, 0]
    lons = np.rad2deg(da_target_time.clon)
    lats = np.rad2deg(da_target_time.clat)
    
    ax.tripcolor(
        lons,
        lats,
        da_target_time.values,
        shading="gouraud",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    is_bottom = (i == len(time_steps) - 1)
    if is_bottom:
        gl = ax.gridlines(
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
            draw_labels=True,
        )
        gl.right_labels = False
        gl.top_labels = False
    ax.spines['geo'].set_visible(False)
    
    if i == 0:
        ax.set_title("Ground Truth", size=10)
    
    ax.annotate(
        pd.Timestamp(time_step).to_pydatetime().strftime('%Y-%m-%d %H:%M'),
        xy=(0.8, 0.05), 
        xycoords='axes fraction',
        fontsize=8,
        ha="right",
        va="bottom",
    )
    
    # Plot predictions in remaining columns
    for j, (name, da_prediction) in enumerate(da_predictions.items(), start=1):
        ax = axes[i, j]
        da_prediction_time = da_prediction.isel(elapsed_forecast_duration=i)
        
        ax.tripcolor(
            lons,
            lats,
            da_prediction_time.values,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.spines['geo'].set_visible(False)
        
        if i == 0:
            ax.set_title(name, size=10)
        
        ax.annotate(
            f"{da_prediction.start_time.dt.strftime('%Y-%m-%d %H:%M:%S').values.item()} + {int(da_prediction.elapsed_forecast_duration.values[i].astype('timedelta64[m]').item().total_seconds() / 60)} minutes",
            xy=(0.8, 0.05), 
            xycoords='axes fraction',
            fontsize=8,
            ha="right",
            va="bottom",
        )

# Add a single colorbar for the entire figure
cbar_ax = fig.add_axes([1., 0.3, 0.02, 0.4])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Temperature (K)", size=12)

plt.tight_layout()
plt.savefig("single_forecast_comparison.png", dpi=300, bbox_inches="tight")
