#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/.venv/bin/python
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=30G
#SBATCH --gres=tmpfs:100G
#SBATCH -t 0:40:00
import xarray as xr
import sys
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pandas as pd
from pathlib import Path

sys.path.append("../src/helpers")
from analysis_helpers import combine_state_features

# Helper: convert xarray / dask-backed values to NumPy (safe for matplotlib)
def _to_numpy(x):
    """Return a numpy array or scalar for x which may be an xarray DataArray, dask or numpy array, or scalar."""
    if isinstance(x, xr.DataArray):
        x = x.values
    # dask arrays have .compute()
    if hasattr(x, "compute"):
        x = x.compute()
    return np.asarray(x)

# Define prediction datasets to compare
prediction_datasets = {
    "Baseline": "../evals/baseline_unroll30.zarr",
    "Baseline + rr": "../evals/rain_mse_boxcox2.zarr",
    # "Baseline + rr": "../evals/rain_mse_boxcox_unroll100.zarr",
    "Baseline + qv": "../evals/qv_unroll30.zarr",
    "Baseline + fluxes": "../evals/lhfl_unroll30.zarr",
    # "Baseline + BT": "../evals/rain_mse_boxcox2.zarr",
    "Surface only": "../evals/sfconly_unroll30.zarr",
    "Bare-minimum": "../evals/bare-minimum_unroll30.zarr",
}

# Load predictions
predictions = {
    name: combine_state_features(xr.open_dataset(path, engine="zarr", chunks="auto"))
    for name, path in prediction_datasets.items()
}

# Map model names (keys from `prediction_datasets`) to precomputed metric files
model_to_metrics = {
    "Baseline + rr": "metrics_rain_mse_boxcox2.nc",
    "Baseline + qv": "metrics_qv.nc",
    "Baseline + fluxes": "metrics_lhfl.nc",
    "Surface only": "metrics_sfconly.nc",
    "Bare-minimum": "metrics_bare-minimum.nc",
    "Baseline": "metrics_baseline.nc"
}
base_dir = Path(__file__).resolve().parent
metrics_datasets = {}
for model_name, fname in model_to_metrics.items():
    # only try to load metrics for models we actually plotted
    if model_name not in predictions:
        continue
    p = base_dir / fname
    if p.exists():
        try:
            metrics_datasets[model_name] = xr.open_dataset(p, chunks="auto")
        except Exception as e:
            print(f"Could not open metrics file {p}: {e}")

# Helper to look up metric values for a given forecast/init and leadtime (hours)
def _lookup_metric(ds, var_name, forecast_time, leadtime):
    """Return metric value (float) or None if not available/mismatch."""
    if ds is None or var_name not in ds:
        return None
    try:
        forecast_np = np.datetime64(pd.Timestamp(forecast_time).to_datetime64())
    except Exception:
        forecast_np = np.datetime64(forecast_time)
    # forecast must exist exactly
    if forecast_np not in ds.coords["forecast"].values:
        return None
    lead_vals = ds["leadtime"].values
    try:
        return float(ds[var_name].sel(forecast=forecast_np, leadtime=leadtime).values)
    except Exception:
        return None

# Load ground truth
ground_truth = xr.open_dataset("../data/experiment/data/datastore.interior.domain03.zarr", engine="zarr")
combined_ground_truth = combine_state_features(ground_truth)

# Extract prediction data arrays for each model
da_predictions = {
    name: pred.t_2m.sel(start_time="2020-02-12T15:30:00").isel(elapsed_forecast_duration=[0,1,2,3,10,19])
    for name, pred in predictions.items()
}

# Use the first prediction to determine times
first_prediction = list(da_predictions.values())[0]
prediction_times = first_prediction.start_time.values + first_prediction.elapsed_forecast_duration.values
da_target = combined_ground_truth.t_2m.sel(time=prediction_times)

assert da_target.time.shape == first_prediction.elapsed_forecast_duration.shape

# Calculate global min/max across all predictions and target
# Ensure values are computed (dask -> numpy scalars) so matplotlib receives plain floats
vals_min = [da_target.min().compute().item()] + [pred.min().compute().item() for pred in da_predictions.values()]
vmin = float(min(vals_min))
vals_max = [da_target.max().compute().item()] + [pred.max().compute().item() for pred in da_predictions.values()]
vmax = float(max(vals_max))


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
            _to_numpy(da_prediction_time),
            shading="gouraud",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.spines['geo'].set_visible(False)
        
        if i == 0:
            ax.set_title(name, size=10)
        
        # Base label (start + lead minutes) — handle dask-backed/xarray scalars
        start_val = _to_numpy(da_prediction.start_time.dt.strftime('%Y-%m-%d %H:%M:%S'))
        start_str = start_val.item() if getattr(start_val, "size", 1) == 1 else str(start_val)
        elapsed_vals = _to_numpy(da_prediction.elapsed_forecast_duration)
        minutes = int(pd.to_timedelta(elapsed_vals[i]).total_seconds() / 60)
        label_text = ""

        # Append RMSE / IQR from precomputed metrics if available for this model / forecast / lead
        metrics_ds = metrics_datasets.get(name)
        if metrics_ds is not None:
            forecast_time_val = _to_numpy(da_prediction.start_time)
            forecast_time = forecast_time_val.item() if getattr(forecast_time_val, "size", 1) == 1 else forecast_time_val[0]
            lead_val = _to_numpy(da_prediction.elapsed_forecast_duration)[i]
            lead_hours = pd.to_timedelta(lead_val).total_seconds() / 3600.0
            rmse_val = _lookup_metric(metrics_ds, "t_2m_rmse", forecast_time, lead_hours)
            iqr_val = _lookup_metric(metrics_ds, "t_2m_iqr", forecast_time, lead_hours)
            print(f"{forecast_time}, {lead_hours}, {rmse_val}, {iqr_val}")
            if rmse_val is not None or iqr_val is not None:
                parts = []
                if rmse_val is not None:
                    parts.append(f"RMSE {rmse_val:.2f} K")
                if iqr_val is not None:
                    parts.append(f"IQR {iqr_val:.2f} K")
                label_text += ",\n".join(parts)

        ax.annotate(
            f"{start_str} + {minutes} minutes",
            xy=(0.8, 0.05),
            xycoords='axes fraction',
            fontsize=8,
            ha="right",
            va="bottom",
        )
        ax.annotate(
            label_text,
            xy=(0.05, 0.9),
            xycoords='axes fraction',
            fontsize=8,
            ha="left",
            va="bottom",
        )
            

# Add a single colorbar for the entire figure
cbar_ax = fig.add_axes([1., 0.3, 0.02, 0.4])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("2m temperature / K", size=12)

plt.tight_layout()
plt.savefig("single_forecast_comparison.png", dpi=300, bbox_inches="tight")
print("Saved figure to single_forecast_comparison.png")
