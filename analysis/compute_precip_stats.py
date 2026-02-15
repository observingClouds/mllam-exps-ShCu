#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/analysis/.venv/bin/python
#SBATCH -n 1
#SBATCH -t 1:00:00
"""
Notebook to compute precipitation statistics to test physical consistency
"""

import pickle
import xarray as xr
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

sys.path.append('../src')
import graph_operators as go

level = 6
graph_path = "../data/multilevel_graph_hierarchy.domain03.connected.pkl"
with open(graph_path, 'rb') as f:
    loaded_levels = pickle.load(f)
G_coarse = loaded_levels[level]['graph']
level_cells = list(G_coarse.nodes.keys())
coarse_positions = loaded_levels[level]['pos']

pred = xr.open_dataset("../evals/rain_mse_boxcox2.zarr", engine="zarr")

reference = False
if reference:
    ground_truth = xr.open_dataset("reference://", storage_options={'fo':"/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/ablation_study/index.interior.boxcoxrain.json"}, engine="zarr")
    t_2m_full = ground_truth.sel(state_feature='t_2m').rename({'grid_index': 'cell'})
    rr_full = ground_truth.sel(state_feature='rain_gsp_rate').rename({'grid_index': 'cell'})
else:
    t_2m_full = pred.sel(state_feature='t_2m').rename({'grid_index': 'cell'})
    rr_full = pred.sel(state_feature='rain_gsp_rate').rename({'grid_index': 'cell'})

# Get dimensions
start_times = pred.start_time.values
forecast_durations = pred.elapsed_forecast_duration.values

# Initialize arrays to store metrics
metrics_shape = (len(start_times), len(forecast_durations))
n_t2m_minima = np.zeros(metrics_shape, dtype=int)
n_rain_maxima = np.zeros(metrics_shape, dtype=int)
n_matching = np.zeros(metrics_shape, dtype=int)
n_total_unique = np.zeros(metrics_shape, dtype=int)
score_from_t2m_arr = np.zeros(metrics_shape, dtype=float)
score_from_rain_arr = np.zeros(metrics_shape, dtype=float)
f1_score_arr = np.zeros(metrics_shape, dtype=float)
jaccard_score_arr = np.zeros(metrics_shape, dtype=float)
mean_score_arr = np.zeros(metrics_shape, dtype=float)

coarsen = level

# Loop over all time dimensions
for i, start_time in enumerate(tqdm(start_times, desc="Processing start times", leave=True)):
    for j, forecast_duration in enumerate(tqdm(forecast_durations, desc="Processing forecast durations", leave=False)):
        # Select data for this time point
        if reference:
            t_2m = t_2m_full.sel(time=start_time+forecast_duration)
            rr = rr_full.sel(time=start_time+forecast_duration)
        else:
            t_2m = t_2m_full.isel(start_time=i, elapsed_forecast_duration=j)
            rr = rr_full.isel(start_time=i, elapsed_forecast_duration=j)
        
        # Coarsen data and set node attributes
        level_data_t_2m = t_2m.state.values.reshape(-1, 4**coarsen).mean(axis=-1)
        assert len(level_data_t_2m) == len(level_cells)
        nx.set_node_attributes(G_coarse, dict(zip(level_cells, level_data_t_2m)), 't_2m')
        
        level_data_rr = rr.state.values.reshape(-1, 4**coarsen).mean(axis=-1)
        assert len(level_data_rr) == len(level_cells)
        nx.set_node_attributes(G_coarse, dict(zip(level_cells, level_data_rr)), 'rain_gsp_rate')
        
        # Find local extrema
        local_minima_t2m = go.find_local_minima(G_coarse, attribute='t_2m')
        local_maxima_rain = go.find_local_maxima(G_coarse, attribute='rain_gsp_rate')
        
        # Calculate overlap
        minima_t2m_set = set(local_minima_t2m)
        maxima_rain_set = set(local_maxima_rain)
        matching_cells = minima_t2m_set.intersection(maxima_rain_set)
        total_unique = len(minima_t2m_set.union(maxima_rain_set))
        
        # Store counts
        n_t2m_minima[i, j] = len(local_minima_t2m)
        n_rain_maxima[i, j] = len(local_maxima_rain)
        n_matching[i, j] = len(matching_cells)
        n_total_unique[i, j] = total_unique
        
        # Calculate scores
        if len(local_minima_t2m) > 0:
            score_from_t2m_arr[i, j] = len(matching_cells) / len(local_minima_t2m)
        
        if len(local_maxima_rain) > 0:
            score_from_rain_arr[i, j] = len(matching_cells) / len(local_maxima_rain)
        
        # F1 score
        if score_from_t2m_arr[i, j] + score_from_rain_arr[i, j] > 0:
            f1_score_arr[i, j] = 2 * (score_from_t2m_arr[i, j] * score_from_rain_arr[i, j]) / (score_from_t2m_arr[i, j] + score_from_rain_arr[i, j])
        
        # Jaccard similarity
        if total_unique > 0:
            jaccard_score_arr[i, j] = len(matching_cells) / total_unique
        
        # Arithmetic mean
        mean_score_arr[i, j] = (score_from_t2m_arr[i, j] + score_from_rain_arr[i, j]) / 2

# Create xarray Dataset with results
metrics_ds = xr.Dataset(
    {
        'n_t2m_minima': (['start_time', 'elapsed_forecast_duration'], n_t2m_minima),
        'n_rain_maxima': (['start_time', 'elapsed_forecast_duration'], n_rain_maxima),
        'n_matching': (['start_time', 'elapsed_forecast_duration'], n_matching),
        'n_total_unique': (['start_time', 'elapsed_forecast_duration'], n_total_unique),
        'score_from_t2m': (['start_time', 'elapsed_forecast_duration'], score_from_t2m_arr),
        'score_from_rain': (['start_time', 'elapsed_forecast_duration'], score_from_rain_arr),
        'f1_score': (['start_time', 'elapsed_forecast_duration'], f1_score_arr),
        'jaccard_score': (['start_time', 'elapsed_forecast_duration'], jaccard_score_arr),
        'mean_score': (['start_time', 'elapsed_forecast_duration'], mean_score_arr),
    },
    coords={
        'start_time': start_times,
        'elapsed_forecast_duration': forecast_durations,
    },
)

# Print summary statistics
print("\n" + "="*60)
print("OVERLAP ANALYSIS: Cold Spots (t_2m min) vs Rain Peaks (rain max)")
print("="*60)
print(f"Computed metrics for {len(start_times)} start times and {len(forecast_durations)} forecast durations")
print(f"\nMean values across all times:")
print(f"  t_2m minima (cold spots):         {metrics_ds.n_t2m_minima.mean().values:.1f}")
print(f"  rain maxima (rain peaks):         {metrics_ds.n_rain_maxima.mean().values:.1f}")
print(f"  matching cells (intersection):    {metrics_ds.n_matching.mean().values:.1f}")
print(f"  total unique extrema (union):     {metrics_ds.n_total_unique.mean().values:.1f}")
print(f"\nMean scores across all times:")
print(f"  Matches / t_2m minima:            {metrics_ds.score_from_t2m.mean().values:.2%}")
print(f"  Matches / rain maxima:            {metrics_ds.score_from_rain.mean().values:.2%}")
print(f"  F1-Score (harmonic mean):         {metrics_ds.f1_score.mean().values:.2%}")
print(f"  Jaccard similarity:               {metrics_ds.jaccard_score.mean().values:.2%}")
print(f"  Arithmetic mean:                  {metrics_ds.mean_score.mean().values:.2%}")
print("="*60)

# Display the dataset
print("\nMetrics Dataset:")
print(metrics_ds)

output_path = "../evals/precip_stats.nc"
metrics_ds.to_netcdf(output_path)
