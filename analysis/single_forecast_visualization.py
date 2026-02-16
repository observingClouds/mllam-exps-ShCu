import xarray as xr
import sys
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pandas as pd

sys.path.append("../src/helpers")
from analysis_helpers import combine_state_features

# Load predictions
predictions = xr.open_dataset("../evals/rain_mse_boxcox_unroll100.zarr", engine="zarr")

# Load ground truth
ground_truth = xr.open_dataset("../data/experiment/data/datastore.interior.domain03.zarr", engine="zarr")

combined_predictions = combine_state_features(predictions)
combined_ground_truth = combine_state_features(ground_truth)

da_prediction = combined_predictions.t_2m.sel(start_time="2020-02-12T15:00:00").isel(elapsed_forecast_duration=slice(0,12, 3)) # "2020-02-12T15:00:00" (flower-ish); "2020-02-18T15:00:00" (sugar-ish in a eastern domain)
prediction_times = da_prediction.start_time.values + da_prediction.elapsed_forecast_duration.values
da_target = combined_ground_truth.t_2m.sel(time=prediction_times)

assert da_target.time.shape == da_prediction.elapsed_forecast_duration.shape

vmin = min(da_prediction.min(), da_target.min())
vmax = max(da_prediction.max(), da_target.max())


# Extract the temporal dimension
time_steps = da_target.time.values

# Create subplots for each time step
fig, axes = plt.subplots(
    len(time_steps),
    2,
    figsize=(13, 4 * len(time_steps)),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# Ensure axes is a 2D array for consistent indexing
if len(time_steps) == 1:
    axes = np.array([axes])

# Plot for each time step
for i, time_step in enumerate(time_steps):
    da_target_time = da_target.isel(time=i)
    da_prediction_time = da_prediction.isel(elapsed_forecast_duration=i)

    for ax, da, title in zip(axes[i], [da_target_time, da_prediction_time], ["Ground Truth", "Prediction"]):
        lons = np.rad2deg(da_target_time.clon)
        lats = np.rad2deg(da_target_time.clat)

        ax.tripcolor(
            lons,
            lats,
            da.values,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        gl = ax.gridlines(
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
            draw_labels=True,
        )
        gl.right_labels = False
        gl.top_labels = False
        if ax is axes[i, 1]:
            gl.left_labels = False

        ax.spines['geo'].set_visible(False)
        if i == 0:
            ax.set_title(f"{title}", size=10)
        
        if ax is axes[i, 0]:
            ax.annotate(
            pd.Timestamp(time_step).to_pydatetime().strftime('%Y-%m-%d %H:%M'),
            xy=(0.95, 0.05), 
            xycoords='axes fraction',
            fontsize=8,
            ha="right",
            va="bottom",
            )

        if ax is axes[i, 1]:
            ax.annotate(
            f"{da_prediction.start_time.dt.strftime('%Y-%m-%d %H:%M:%S').values.item()} + {int(da_prediction.elapsed_forecast_duration.values[i].astype('timedelta64[m]').item().total_seconds() / 60)} minutes",
            xy=(0.95, 0.05), 
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