#!/leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/.venv/bin/python
"""
Visualize metrics from pre-computed metric files.

Usage:
    python visualize_metrics.py <metric_file1> <metric_file2> ...
    
Example:
    python visualize_metrics.py metrics_baseline.nc metrics_sfconly.nc
"""
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Variable units
UNITS = {
    't_2m': 'K',
    'u_10m': 'm/s',
    'v_10m': 'm/s',
    'tot_prec': 'kg m-2',
}


def plot_rmse_timeseries(ax, combined_datasets_dict, variable, percentiles=[25, 75]):
    """Plot RMSE time series with median and percentiles for multiple datasets."""
    colors = sns.color_palette("husl", len(combined_datasets_dict))
    
    for idx, (dataset_name, combined_ds) in enumerate(combined_datasets_dict.items()):
        rmse_var_name = f"{variable}_rmse"
        if rmse_var_name not in combined_ds.data_vars:
            continue
            
        rmse_var = combined_ds[rmse_var_name]
        color = colors[idx]
        
        # Calculate median and percentiles
        rmse_median = rmse_var.quantile(0.5, dim='forecast')
        rmse_percentiles = {p: rmse_var.quantile(p / 100, dim='forecast') for p in percentiles}
        
        leadtime = rmse_median.leadtime.values
        
        ax.plot(leadtime, rmse_median, color=color, label=f'{dataset_name}', linewidth=2)
        ax.fill_between(leadtime, rmse_percentiles[min(percentiles)], 
                        rmse_percentiles[max(percentiles)], color=color, alpha=0.2)
    
    ax.set_xlabel('Lead Time / h')
    ax.set_ylabel(f'RMSE ({variable})')
    ax.legend(loc='lower right')
    ax.grid()


def plot_iqr_timeseries(ax, combined_datasets_dict, variable):
    """Plot IQR time series for multiple datasets."""
    colors = sns.color_palette('husl', len(combined_datasets_dict))
    
    for idx, (dataset_name, combined_ds) in enumerate(combined_datasets_dict.items()):
        iqr_var_name = f"{variable}_iqr"
        if iqr_var_name not in combined_ds.data_vars:
            continue
            
        iqr_var = combined_ds[iqr_var_name]
        color = colors[idx]
        
        # Average over extra dimensions
        extra_dims = [dim for dim in iqr_var.dims if dim != 'leadtime']
        if extra_dims:
            iqr_var = iqr_var.mean(dim=extra_dims)
        
        leadtime = iqr_var.leadtime.values
        ax.plot(leadtime, iqr_var, color=color, label=f'{dataset_name}', linewidth=2)
    
    ax.set_xlabel('Lead Time / h')
    ax.set_ylabel(f'IQR ({variable})')
    ax.legend(loc='lower right')
    ax.grid()


def plot_quantile_score_timeseries(ax, combined_datasets_dict, variable, percentiles=[25, 75]):
    """Plot quantile score time series with median and percentiles for multiple datasets."""
    colors = sns.color_palette("husl", len(combined_datasets_dict))
    
    for idx, (dataset_name, combined_ds) in enumerate(combined_datasets_dict.items()):
        qs_var_name = f"{variable}_quantile_score"
        if qs_var_name not in combined_ds.data_vars:
            continue
            
        qs_var = combined_ds[qs_var_name]
        color = colors[idx]
        
        # Calculate median and percentiles
        qs_median = qs_var.quantile(0.5, dim='forecast')
        qs_percentiles = {p: qs_var.quantile(p / 100, dim='forecast') for p in percentiles}
        
        leadtime = qs_median.leadtime.values
        
        ax.plot(leadtime, qs_median, color=color, label=f'{dataset_name}', linewidth=2)
        ax.fill_between(leadtime, qs_percentiles[min(percentiles)], 
                        qs_percentiles[max(percentiles)], color=color, alpha=0.2)
    
    ax.set_xlabel('Lead Time / h')
    ax.set_ylabel(f'75th Quantile Score ({variable})')
    ax.legend(loc='lower right')
    ax.grid()


def load_metric_datasets(metric_files):
    """Load all metric datasets from files."""
    datasets = {}
    for file_path in metric_files:
        path = Path(file_path)
        if not path.exists():
            logging.warning(f"File not found: {file_path}, skipping...")
            continue
        
        # Extract name from filename (remove metrics_ prefix and .nc suffix)
        name = path.stem.replace('metrics_', '')
        
        logging.info(f"Loading {name} from {file_path}")
        datasets[name] = xr.open_dataset(file_path)
    
    return datasets


def extract_variables(datasets):
    """Extract unique base variable names from all datasets."""
    all_variables = set()
    for ds in datasets.values():
        for var_name in ds.data_vars:
            if var_name.endswith('_rmse'):
                base_var = var_name[:-5]
                all_variables.add(base_var)
    return sorted(all_variables)


def create_plots(datasets, variables, output_dir="."):
    """Create all visualization plots."""
    output_dir = Path(output_dir)
    
    for var in variables:
        logging.info(f"Creating plots for {var}...")
        
        # Filter datasets that have this variable
        datasets_with_var_rmse = {name: ds for name, ds in datasets.items() 
                                   if f"{var}_rmse" in ds.data_vars}
        datasets_with_var_iqr = {name: ds for name, ds in datasets.items() 
                                 if f"{var}_iqr" in ds.data_vars}
        datasets_with_var_qs = {name: ds for name, ds in datasets.items() 
                                if f"{var}_quantile_score" in ds.data_vars}
        
        unit = UNITS.get(var, '')
        
        # RMSE plot
        if datasets_with_var_rmse:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            plot_rmse_timeseries(ax, datasets_with_var_rmse, var)
            ax.set_xlim(1/6, 2.5)
            ax.set_ylabel(f'RMSE({var})' + (f' / {unit}' if unit else ''))
            ax.grid(axis='x', which='major', linestyle='--', color='grey', alpha=0.7)
            sns.despine(bottom=True)
            plt.tight_layout()
            filename = output_dir / f"rmse_timeseries_{var}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            logging.info(f"Saved {filename}")
        
        # IQR plot
        if datasets_with_var_iqr:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            plot_iqr_timeseries(ax, datasets_with_var_iqr, var)
            ax.set_xlim(1/6, 2.5)
            ax.set_ylabel(f'IQR({var})' + (f' / {unit}' if unit else ''))
            ax.grid(axis='x', which='major', linestyle='--', color='grey', alpha=0.7)
            sns.despine(bottom=True)
            plt.tight_layout()
            filename = output_dir / f"iqr_timeseries_{var}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            logging.info(f"Saved {filename}")
        
        # Quantile score plot
        if datasets_with_var_qs:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            plot_quantile_score_timeseries(ax, datasets_with_var_qs, var)
            ax.set_xlim(1/6, 2.5)
            ax.set_ylabel(f'75th QS({var})' + (f' / {unit}' if unit else ''))
            ax.grid(axis='x', which='major', linestyle='--', color='grey', alpha=0.7)
            sns.despine(bottom=True)
            plt.tight_layout()
            filename = output_dir / f"quantile_score_timeseries_{var}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
            logging.info(f"Saved {filename}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_metrics.py <metric_file1> <metric_file2> ...")
        print("Example: python visualize_metrics.py metrics_baseline.nc metrics_sfconly.nc")
        sys.exit(1)
    
    metric_files = sys.argv[1:]
    
    # Load datasets
    datasets = load_metric_datasets(metric_files)
    
    if not datasets:
        logging.error("No valid metric files loaded!")
        sys.exit(1)
    
    logging.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Extract variables
    variables = extract_variables(datasets)
    logging.info(f"Found variables: {variables}")
    
    # Create plots
    create_plots(datasets, variables)
    
    logging.info("All plots created successfully!")


if __name__ == "__main__":
    main()
