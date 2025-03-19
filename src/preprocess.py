from intake import open_catalog
from loguru import logger
import xarray as xr
import pandas as pd
import numpy as np
import argparse
from dask.diagnostics import ProgressBar

if __name__ == "__main__":
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
    parser = argparse.ArgumentParser(description="Process and save ICON-LES data.")
    parser.add_argument("--output", type=str, required=True, help="Output file path for the Zarr dataset.", default="/dcai/projects/cu_0003/data/sources/icon/ICON-DOM02.v1.zarr")
    args = parser.parse_args()

    with ProgressBar():
        ds.to_zarr(args.output, mode="w")