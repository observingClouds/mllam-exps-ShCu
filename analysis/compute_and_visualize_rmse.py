#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/.venv/bin/python
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=20G
#SBATCH --partition=lrd_all_serial
#SBATCH -t 3:30:00
import xarray as xr
import pandas as pd
import datetime as dt
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scores.continuous as scc_cont
import sys
import logging
from dask.distributed import LocalCluster

if __name__ == "__main__":
    # Configure logging with timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='20GB')
    client = cluster.get_client()
    logging.info(f"Dask dashboard available at: {client.dashboard_link}")

    frequency = "1h"  # Frequency of forecasts
    length_of_forecast = 20  # Length of forecast in time steps

    sys.path.append("../src/helpers")
    from analysis_helpers import combine_state_features

    # Define prediction datasets to compare
    prediction_datasets = {
        'bare-minimum': '../evals/bare-minimum_unroll30.zarr',
        # 'baseline': '../evals/baseline_unroll30.zarr',
        # 'lhfl': '../evals/lhfl_unroll30.zarr',
        # 'lw': '../evals/lw_unroll30.zarr',
        # 'qv': '../evals/qv_unroll30.zarr',
        'sfconly': '../evals/sfconly_unroll30.zarr',
    }

    # Load ground truth
    ground_truth_ds = xr.open_dataset("../data/experiment/data/datastore.interior.domain03.zarr", engine="zarr", chunks='auto')
    combined_ground_truth = combine_state_features(ground_truth_ds)

    # Process each prediction dataset - store all metrics in combined datasets
    combined_results = {}
    for dataset_name, dataset_path in prediction_datasets.items():
        logging.info(f"Processing {dataset_name}...")
        
        try:
            # Load predictions
            predictions = xr.open_dataset(dataset_path, engine="zarr", chunks='auto').isel(elapsed_forecast_duration=slice(0, length_of_forecast))
        except Exception as e:
            logging.error(f"Could not load {dataset_name} from {dataset_path}: {e}")
            continue
        
        combined_predictions = combine_state_features(predictions)
        
        pcollection = []
        rcollection = []
        for forecast, init_time in enumerate(tqdm.tqdm(pd.date_range(
                start=predictions.start_time.min().values,
                end=predictions.start_time.max().values,
                freq=frequency,
            ), desc=f"{dataset_name}")):
            # Extract the forecast time slice
            try:
                ground_truth = combined_ground_truth.sel(time=slice(init_time+dt.timedelta(minutes=10), init_time + dt.timedelta(minutes=10*length_of_forecast)))
                preds_slice = combined_predictions.sel(start_time=init_time).isel(elapsed_forecast_duration=slice(0, length_of_forecast))
            except KeyError:
                logging.warning(f"KeyError for init_time: {init_time}, skipping...")
                continue
            assert len(preds_slice.time) == length_of_forecast, f"Forecast length mismatch: {len(preds_slice.time)} != {length_of_forecast}"
            assert len(ground_truth.time) == length_of_forecast, f"Ground truth length mismatch: {len(ground_truth.time)} != {length_of_forecast}"

            preds = preds_slice.rename({'elapsed_forecast_duration': 'leadtime',}).expand_dims('forecast').drop_vars(['time', 'start_time', 'state_feature'])
            reference = ground_truth.rename({'time': 'leadtime',}).expand_dims('forecast').drop_vars(['state_feature', 'state_feature_long_name', 'state_feature_units', 'state_feature_source_dataset'])
            reference = reference.assign_coords({'leadtime': pd.date_range(start=init_time+dt.timedelta(minutes=10), periods=length_of_forecast, freq='10min') - init_time})

            pcollection.append(preds)
            rcollection.append(reference)
        
        if len(pcollection) == 0:
            logging.warning(f"No valid forecasts found for {dataset_name}, skipping...")
            continue
        
        logging.info(f"Concatenating forecasts for {dataset_name}...")
        p = xr.concat(pcollection, dim='forecast')
        r = xr.concat(rcollection, dim='forecast')

        # Calculate all metrics at once
        logging.info(f"Calculating all metrics for {dataset_name}...")
        
        # RMSE
        rmse = scc_cont.rmse(p, r, preserve_dims=['leadtime', 'forecast'])
        rmse = rmse.assign_coords(leadtime=rmse.leadtime.dt.total_seconds() / (60 * 60))
        
        # Quantile score
        quantile_score = scc_cont.quantile_score(p, r, preserve_dims=['leadtime', 'forecast'], alpha=0.75)
        quantile_score = quantile_score.assign_coords(leadtime=quantile_score.leadtime.dt.total_seconds() / (60 * 60))
        
        # IQR
        q25 = p.quantile(0.25, dim=['forecast', 'grid_index'])
        q75 = p.quantile(0.75, dim=['forecast', 'grid_index'])
        iqr = q75 - q25
        iqr = iqr.assign_coords(leadtime=iqr.leadtime.dt.total_seconds() / (60 * 60))
        
        # Combine all metrics into a single dataset
        # Create separate datasets with metric suffix
        rmse_ds = xr.Dataset({f"{var}_rmse": rmse[var] for var in rmse.data_vars})
        qs_ds = xr.Dataset({f"{var}_quantile_score": quantile_score[var] for var in quantile_score.data_vars})
        iqr_ds = xr.Dataset({f"{var}_iqr": iqr[var] for var in iqr.data_vars})
        
        # Merge all into one dataset
        combined = xr.merge([rmse_ds, qs_ds, iqr_ds])
        combined_results[dataset_name] = combined
        
        logging.info(f"Completed {dataset_name}: {len(combined.data_vars)} metric variables")

    # Combine all datasets with proper 'dataset' dimension
    datasets_with_dim = []
    for dataset_name, ds in combined_results.items():
        ds_expanded = ds.expand_dims({'dataset': [dataset_name]})
        datasets_with_dim.append(ds_expanded)
    
    combined_metrics = xr.concat(datasets_with_dim, dim='dataset')
    combined_metrics.to_netcdf("combined_metrics.nc")
    logging.info("Combined metrics saved to combined_metrics.nc")
    # Extract base variables (without metric suffix) for plotting
    all_variables = set()
    for ds in combined_results.values():
        for var_name in ds.data_vars:
            # Extract base variable name (before _rmse, _iqr, _quantile_score)
            if var_name.endswith('_rmse'):
                base_var = var_name[:-5]
                all_variables.add(base_var)

    logging.info(f"Found base variables across all datasets: {sorted(all_variables)}")

    def plot_rmse_timeseries(ax, combined_datasets_dict, variable, percentiles=[25, 75]):
        """
        Plot RMSE time series with median and percentiles for multiple datasets.

        Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        combined_datasets_dict (dict): Dictionary with dataset names as keys and combined metric datasets as values.
        variable (str): The base variable name to plot.
        percentiles (list): List of percentiles to plot.
        """
        colors = sns.color_palette("husl", len(combined_datasets_dict))
        
        for idx, (dataset_name, combined_ds) in enumerate(combined_datasets_dict.items()):
            rmse_var_name = f"{variable}_rmse"
            # Skip if variable not in this dataset
            if rmse_var_name not in combined_ds.data_vars:
                continue
                
            rmse_var = combined_ds[rmse_var_name]
            color = colors[idx]
            
            # Calculate median and percentiles
            rmse_median = rmse_var.quantile(0.5, dim='forecast')
            rmse_percentiles = {p: rmse_var.quantile(p / 100, dim='forecast') for p in percentiles}

            # Prepare data for plotting
            leadtime = rmse_median.leadtime.values

            # Plotting
            ax.plot(leadtime, rmse_median, color=color, label=f'{dataset_name}', linewidth=2)
            ax.fill_between(leadtime, rmse_percentiles[min(percentiles)], rmse_percentiles[max(percentiles)], 
                        color=color, alpha=0.2)

        ax.set_xlabel('Lead Time / h')
        ax.set_ylabel(f'RMSE ({variable})')
        ax.legend(loc='lower right')
        ax.grid()

    def plot_iqr_timeseries(ax, combined_datasets_dict, variable):
        """
        Plot IQR time series for multiple datasets.

        Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        combined_datasets_dict (dict): Dictionary with dataset names as keys and combined metric datasets as values.
        variable (str): The base variable name to plot.
        """
        colors = sns.color_palette('husl', len(combined_datasets_dict))
        
        for idx, (dataset_name, combined_ds) in enumerate(combined_datasets_dict.items()):
            iqr_var_name = f"{variable}_iqr"
            # Skip if variable not in this dataset
            if iqr_var_name not in combined_ds.data_vars:
                continue
                
            iqr_var = combined_ds[iqr_var_name]
            color = colors[idx]
            
            # Ensure we only have leadtime dimension by averaging over any extra dimensions
            extra_dims = [dim for dim in iqr_var.dims if dim != 'leadtime']
            if extra_dims:
                iqr_var = iqr_var.mean(dim=extra_dims)
            
            # Prepare data for plotting
            leadtime = iqr_var.leadtime.values

            # Plotting
            ax.plot(leadtime, iqr_var, color=color, label=f'{dataset_name}', linewidth=2)

        ax.set_xlabel('Lead Time / h')
        ax.set_ylabel(f'IQR ({variable})')
        ax.legend(loc='lower right')
        ax.grid()

    def plot_quantile_score_timeseries(ax, combined_datasets_dict, variable, percentiles=[25, 75]):
        """
        Plot quantile score time series with median and percentiles for multiple datasets.

        Parameters:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        combined_datasets_dict (dict): Dictionary with dataset names as keys and combined metric datasets as values.
        variable (str): The base variable name to plot.
        percentiles (list): List of percentiles to plot.
        """
        colors = sns.color_palette("husl", len(combined_datasets_dict))
        
        for idx, (dataset_name, combined_ds) in enumerate(combined_datasets_dict.items()):
            qs_var_name = f"{variable}_quantile_score"
            # Skip if variable not in this dataset
            if qs_var_name not in combined_ds.data_vars:
                continue
                
            qs_var = combined_ds[qs_var_name]
            color = colors[idx]
            
            # Calculate median and percentiles
            qs_median = qs_var.quantile(0.5, dim='forecast')
            qs_percentiles = {p: qs_var.quantile(p / 100, dim='forecast') for p in percentiles}

            # Prepare data for plotting
            leadtime = qs_median.leadtime.values

            # Plotting
            ax.plot(leadtime, qs_median, color=color, label=f'{dataset_name}', linewidth=2)
            ax.fill_between(leadtime, qs_percentiles[min(percentiles)], qs_percentiles[max(percentiles)], 
                        color=color, alpha=0.2)

        ax.set_xlabel('Lead Time / h')
        ax.set_ylabel(f'75th Quantile Score ({variable})')
        ax.legend(loc='lower right')
        ax.grid()

    units = {
        't_2m': 'K',
        'u_10m': 'm/s',
        'v_10m': 'm/s',
        'tot_prec': 'kg m-2',
    }

    # Create RMSE comparison plots - one figure per variable
    sorted_variables = sorted(all_variables)
    for var in sorted_variables:
        # Filter datasets that have this variable
        datasets_with_var = {name: ds for name, ds in combined_results.items() if f"{var}_rmse" in ds.data_vars}
        
        if not datasets_with_var:
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_rmse_timeseries(ax, datasets_with_var, var)
        
        ax.set_xlim(1/6, 2.5)
        unit = units.get(var, '')
        ax.set_ylabel(f'RMSE({var})' + (f' / {unit}' if unit else ''))
        ax.grid(axis='x', which='major', linestyle='--', color='grey', alpha=0.7)
        
        sns.despine(bottom=True)
        plt.tight_layout()
        filename = f"rmse_timeseries_{var}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        logging.info(f"Plot saved as {filename}")

    # Create IQR comparison plots - one figure per variable
    for var in sorted_variables:
        # Filter datasets that have this variable
        datasets_with_var = {name: ds for name, ds in combined_results.items() if f"{var}_iqr" in ds.data_vars}
        
        if not datasets_with_var:
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_iqr_timeseries(ax, datasets_with_var, var)
        
        ax.set_xlim(1/6, 2.5)
        unit = units.get(var, '')
        ax.set_ylabel(f'IQR({var})' + (f' / {unit}' if unit else ''))
        ax.grid(axis='x', which='major', linestyle='--', color='grey', alpha=0.7)
        
        sns.despine(bottom=True)
        plt.tight_layout()
        filename = f"iqr_timeseries_{var}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        logging.info(f"Plot saved as {filename}")

    # Create quantile score comparison plots - one figure per variable
    for var in sorted_variables:
        # Filter datasets that have this variable
        datasets_with_var = {name: ds for name, ds in combined_results.items() if f"{var}_quantile_score" in ds.data_vars}
        
        if not datasets_with_var:
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_quantile_score_timeseries(ax, datasets_with_var, var)
        
        ax.set_xlim(1/6, 2.5)
        unit = units.get(var, '')
        ax.set_ylabel(f'75th QS({var})' + (f' / {unit}' if unit else ''))
        ax.grid(axis='x', which='major', linestyle='--', color='grey', alpha=0.7)
        
        sns.despine(bottom=True)
        plt.tight_layout()
        filename = f"quantile_score_timeseries_{var}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        logging.info(f"Plot saved as {filename}")

    logging.info("All plots saved successfully!")