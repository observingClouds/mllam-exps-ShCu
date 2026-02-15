#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/analysis/.venv/bin/python
#SBATCH -n 1
#SBATCH -t 4:00:00
import pickle
import xarray as xr
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

sys.path.append('../src')
import graph_operators as go

# Configuration
levels = [3,4,5,6]
graph_path = "../data/multilevel_graph_hierarchy.domain03.connected.pkl"
n_random_samples = 1  # Number of random samples to average over
random_seed = 42

# Load graph hierarchy
with open(graph_path, 'rb') as f:
    loaded_levels = pickle.load(f)

# Load prediction data
pred = xr.open_dataset("../evals/rain_mse_boxcox2.zarr", engine="zarr")

# Load ground truth reference
ground_truth = xr.open_dataset(
    "reference://", 
    storage_options={'fo': "/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/ablation_study/index.interior.boxcoxrain.json"}, 
    engine="zarr"
)
t_2m_full = ground_truth.sel(state_feature='t_2m').rename({'grid_index': 'cell'})
rr_full = ground_truth.sel(state_feature='rain_gsp_rate').rename({'grid_index': 'cell'})

# Get dimensions
start_times = pred.start_time.values
forecast_durations = pred.elapsed_forecast_duration.values

# Initialize arrays to store random baseline metrics
metrics_shape = (len(levels), len(start_times), len(forecast_durations))
random_n_matching = np.zeros(metrics_shape, dtype=float)
random_score_from_t2m = np.zeros(metrics_shape, dtype=float)
random_score_from_rain = np.zeros(metrics_shape, dtype=float)
random_f1_score = np.zeros(metrics_shape, dtype=float)
random_jaccard_score = np.zeros(metrics_shape, dtype=float)
random_mean_score = np.zeros(metrics_shape, dtype=float)

# Set random seed for reproducibility
np.random.seed(random_seed)

# Loop over all levels and time dimensions
for level_idx, level in enumerate(levels):
    G_coarse = loaded_levels[level]['graph']
    level_cells = list(G_coarse.nodes.keys())
    n_cells = len(level_cells)
    coarsen = level

    for i, start_time in enumerate(tqdm(start_times, desc=f"Processing start times (level={level})", leave=True)):
        for j, forecast_duration in enumerate(tqdm(forecast_durations, desc="Processing forecast durations", leave=False)):
            # Select data for this time point
            t_2m = t_2m_full.sel(time=start_time + forecast_duration)
            rr = rr_full.sel(time=start_time + forecast_duration)

            # Coarsen data and set node attributes
            level_data_t_2m = t_2m.state.values.reshape(-1, 4**coarsen).mean(axis=-1)
            assert len(level_data_t_2m) == len(level_cells)
            nx.set_node_attributes(G_coarse, dict(zip(level_cells, level_data_t_2m)), 't_2m')

            level_data_rr = rr.state.values.reshape(-1, 4**coarsen).mean(axis=-1)
            assert len(level_data_rr) == len(level_cells)
            nx.set_node_attributes(G_coarse, dict(zip(level_cells, level_data_rr)), 'rain_gsp_rate')

            # Find ACTUAL local extrema (to get the counts)
            local_minima_t2m = go.find_local_minima(G_coarse, attribute='t_2m')
            local_maxima_rain = go.find_local_maxima(G_coarse, attribute='rain_gsp_rate')
            
            n_minima = len(local_minima_t2m)
            n_maxima = len(local_maxima_rain)

            # Repeat random assignment multiple times and average
            sample_matching = []
            sample_score_from_t2m = []
            sample_score_from_rain = []
            sample_f1 = []
            sample_jaccard = []
            sample_mean = []

            for _ in range(n_random_samples):
                # Randomly assign extrema to nodes (without replacement)
                random_minima = set(np.random.choice(level_cells, size=n_minima, replace=False))
                random_maxima = set(np.random.choice(level_cells, size=n_maxima, replace=False))

                # Calculate overlap metrics for this random sample
                matching_cells = random_minima.intersection(random_maxima)
                total_unique = len(random_minima.union(random_maxima))
                n_match = len(matching_cells)

                sample_matching.append(n_match)

                # Calculate scores
                score_t2m = n_match / n_minima if n_minima > 0 else 0
                score_rain = n_match / n_maxima if n_maxima > 0 else 0
                sample_score_from_t2m.append(score_t2m)
                sample_score_from_rain.append(score_rain)

                # F1 score
                if score_t2m + score_rain > 0:
                    f1 = 2 * (score_t2m * score_rain) / (score_t2m + score_rain)
                else:
                    f1 = 0
                sample_f1.append(f1)

                # Jaccard similarity
                jaccard = n_match / total_unique if total_unique > 0 else 0
                sample_jaccard.append(jaccard)

                # Arithmetic mean
                mean_score = (score_t2m + score_rain) / 2
                sample_mean.append(mean_score)

            # Store averaged metrics
            random_n_matching[level_idx, i, j] = np.mean(sample_matching)
            random_score_from_t2m[level_idx, i, j] = np.mean(sample_score_from_t2m)
            random_score_from_rain[level_idx, i, j] = np.mean(sample_score_from_rain)
            random_f1_score[level_idx, i, j] = np.mean(sample_f1)
            random_jaccard_score[level_idx, i, j] = np.mean(sample_jaccard)
            random_mean_score[level_idx, i, j] = np.mean(sample_mean)

# Create xarray Dataset with random baseline results
random_metrics_ds = xr.Dataset(
    {
        'n_matching': (['level', 'start_time', 'elapsed_forecast_duration'], random_n_matching),
        'score_from_t2m': (['level', 'start_time', 'elapsed_forecast_duration'], random_score_from_t2m),
        'score_from_rain': (['level', 'start_time', 'elapsed_forecast_duration'], random_score_from_rain),
        'f1_score': (['level', 'start_time', 'elapsed_forecast_duration'], random_f1_score),
        'jaccard_score': (['level', 'start_time', 'elapsed_forecast_duration'], random_jaccard_score),
        'mean_score': (['level', 'start_time', 'elapsed_forecast_duration'], random_mean_score),
    },
    coords={
        'level': np.array(levels),
        'start_time': start_times,
        'elapsed_forecast_duration': forecast_durations,
    },
    attrs={
        'n_random_samples': n_random_samples,
        'random_seed': random_seed,
        'description': 'Random baseline metrics with extrema randomly assigned to grid nodes'
    }
)

# Print summary statistics
print("\n" + "="*60)
print("RANDOM BASELINE: Randomly Assigned Extrema")
print("="*60)
print(f"Random samples per time point: {n_random_samples}")
print(f"Computed metrics for {len(levels)} levels, {len(start_times)} start times, and {len(forecast_durations)} forecast durations")
print(f"\nMean random baseline scores across all times:")
print(f"  Matches / t_2m minima:            {random_metrics_ds.score_from_t2m.mean().values:.2%}")
print(f"  Matches / rain maxima:            {random_metrics_ds.score_from_rain.mean().values:.2%}")
print(f"  F1-Score (harmonic mean):         {random_metrics_ds.f1_score.mean().values:.2%}")
print(f"  Jaccard similarity:               {random_metrics_ds.jaccard_score.mean().values:.2%}")
print(f"  Arithmetic mean:                  {random_metrics_ds.mean_score.mean().values:.2%}")
print("="*60)

# Display the dataset
print("\nRandom Baseline Metrics Dataset:")
print(random_metrics_ds)

# Save to file
output_path = "../evals/precip_stats_multilevel.random_ref.nc"
random_metrics_ds.to_netcdf(output_path)
print(f"\nSaved to {output_path}")