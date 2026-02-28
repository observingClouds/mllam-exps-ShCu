#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/.venv/bin/python
"""
Compute evaluation metrics (RMSE, quantile score, IQR) for a single prediction dataset.

Usage:
    python compute_metrics.py <prediction_path> <output_name>
    
Example:
    python compute_metrics.py ../evals/baseline_unroll30.zarr baseline
"""
import xarray as xr
import pandas as pd
import datetime as dt
import tqdm
import sys
import logging
from pathlib import Path
import scores.continuous as scc_cont
from dask.distributed import LocalCluster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def combine_state_features(ds):
    """Combine state features from helpers."""
    sys.path.append("../src/helpers")
    from analysis_helpers import combine_state_features as csf
    return csf(ds)

def compute_metrics(prediction_path, ground_truth_path, frequency="1h", length_of_forecast=20):
    """
    Compute metrics for a single prediction dataset.
    
    Parameters:
    -----------
    prediction_path : str
        Path to prediction zarr dataset
    ground_truth_path : str
        Path to ground truth zarr dataset
    frequency : str
        Frequency of forecasts (default: "1h")
    length_of_forecast : int
        Number of time steps in forecast (default: 20)
        
    Returns:
    --------
    xarray.Dataset
        Combined metrics dataset with RMSE, quantile score, and IQR
    """
    logging.info(f"Loading ground truth from {ground_truth_path}")
    ground_truth_ds = xr.open_dataset(ground_truth_path, engine="zarr", chunks='auto')
    combined_ground_truth = combine_state_features(ground_truth_ds)
    
    logging.info(f"Loading predictions from {prediction_path}")
    predictions = xr.open_dataset(prediction_path, engine="zarr", chunks='auto').isel(
        elapsed_forecast_duration=slice(0, length_of_forecast)
    )
    combined_predictions = combine_state_features(predictions)
    
    # Collect forecast pairs
    pcollection = []
    rcollection = []
    
    logging.info("Processing forecasts...")
    for forecast, init_time in enumerate(tqdm.tqdm(pd.date_range(
            start=predictions.start_time.min().values,
            end=predictions.start_time.max().values,
            freq=frequency,
        ))):
        try:
            ground_truth = combined_ground_truth.sel(
                time=slice(
                    init_time + dt.timedelta(minutes=10), 
                    init_time + dt.timedelta(minutes=10*length_of_forecast)
                )
            )
            preds_slice = combined_predictions.sel(start_time=init_time).isel(
                elapsed_forecast_duration=slice(0, length_of_forecast)
            )
        except KeyError:
            logging.warning(f"KeyError for init_time: {init_time}, skipping...")
            continue
        
        # Validate lengths
        if len(preds_slice.time) != length_of_forecast or len(ground_truth.time) != length_of_forecast:
            logging.warning(f"Length mismatch at {init_time}, skipping...")
            continue
        
        # Prepare predictions
        preds = preds_slice.rename({'elapsed_forecast_duration': 'leadtime'}).expand_dims({'forecast': [init_time]})
        preds = preds.drop_vars(['time', 'start_time', 'state_feature'])
        
        # Prepare reference
        reference = ground_truth.rename({'time': 'leadtime'}).expand_dims({'forecast': [init_time]})
        reference = reference.drop_vars([
            'state_feature', 'state_feature_long_name', 
            'state_feature_units', 'state_feature_source_dataset'
        ])
        reference = reference.assign_coords({
            'leadtime': pd.date_range(
                start=init_time + dt.timedelta(minutes=10), 
                periods=length_of_forecast, 
                freq='10min'
            ) - init_time
        })
        
        pcollection.append(preds)
        rcollection.append(reference)
    
    if len(pcollection) == 0:
        raise ValueError("No valid forecasts found!")
    
    logging.info("Concatenating forecasts...")
    p = xr.concat(pcollection, dim='forecast')
    r = xr.concat(rcollection, dim='forecast')
    
    logging.info("Calculating metrics...")
    
    # RMSE
    rmse = scc_cont.rmse(p, r, preserve_dims=['leadtime', 'forecast'])
    rmse = rmse.assign_coords(leadtime=rmse.leadtime.dt.total_seconds() / (60 * 60))
    
    # Quantile score
    quantile_score = scc_cont.quantile_score(p, r, preserve_dims=['leadtime', 'forecast'], alpha=0.75)
    quantile_score = quantile_score.assign_coords(leadtime=quantile_score.leadtime.dt.total_seconds() / (60 * 60))
    
    # IQR
    q25 = p.quantile(0.25, dim=['grid_index'])
    q75 = p.quantile(0.75, dim=['grid_index'])
    iqr = q75 - q25
    iqr = iqr.assign_coords(leadtime=iqr.leadtime.dt.total_seconds() / (60 * 60))
    
    # Combine metrics
    rmse_ds = xr.Dataset({f"{var}_rmse": rmse[var] for var in rmse.data_vars})
    qs_ds = xr.Dataset({f"{var}_quantile_score": quantile_score[var] for var in quantile_score.data_vars})
    iqr_ds = xr.Dataset({f"{var}_iqr": iqr[var] for var in iqr.data_vars})
    
    combined = xr.merge([rmse_ds, qs_ds, iqr_ds])
    
    logging.info(f"Computed {len(combined.data_vars)} metric variables")
    return combined


def main():
    if len(sys.argv) != 3:
        print("Usage: python compute_metrics.py <prediction_path> <output_name>")
        print("Example: python compute_metrics.py ../evals/baseline_unroll30.zarr baseline")
        sys.exit(1)
    
    prediction_path = sys.argv[1]
    output_name = sys.argv[2]
    
    # Default ground truth path
    ground_truth_path = "../data/experiment/data/datastore.interior.domain03.zarr"
    
    # Compute metrics
    cluster = LocalCluster()
    client = cluster.get_client()
    logging.info(f"Dask dashboard available at: {client.dashboard_link}")
    metrics = compute_metrics(prediction_path, ground_truth_path)
    
    # Save metrics
    output_file = f"metrics_{output_name}.nc"
    logging.info(f"Saving metrics to {output_file}")
    metrics.to_netcdf(output_file)
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
