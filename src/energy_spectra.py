import numpy as np
import matplotlib.pyplot as plt

DPI = 300
COLORS = {
    "gt": "#000000",  # Black
    "ml": "#E69F00",  # Orange
    "nwp": "#56B4E9",  # Light blue
    "error": "#CC79A7",  # Pink
}

# Line styles and markers for accessibility
LINE_STYLES = {
    "gt": ("solid", "o"),
    "ml": ("solid", "s"),
    "nwp": ("dotted", "^"),
    "ml_tmp": ("dashed", "x"),
}

VARIABLE_UNITS = {
    # Surface and near-surface variables
    "t_2m": "K",

}

def calculate_energy_spectra(data):
    """Calculate the energy spectra of the given data using 2D FFT.

    Parameters
    ----------
    data : xarray.DataArray
        The data for which the energy spectra should be calculated.
        Expected dimensions must include 'x' and 'y', other dimensions will be handled automatically.

    Returns
    -------
    wavenumber : np.ndarray
        The isotropic wavenumbers.
    power : np.ndarray
        The power spectrum averaged over all non-spatial dimensions.
    effective_resolution : float
        The effective resolution of the model.
    """
    # Get grid spacing in meters
    dx = abs(float(data.x[1] - data.x[0]))
    dy = abs(float(data.y[1] - data.y[0]))

    # Identify spatial dimensions
    spatial_dims = ["y", "x"]

    # Move spatial dimensions to the end
    other_dims = [dim for dim in data.dims if dim not in spatial_dims]
    var_data = data.transpose(*other_dims, *spatial_dims).values

    # Reshape the array to combine all non-spatial dimensions
    shape = var_data.shape
    ny, nx = shape[-2:]

    if len(shape) > 2:
        var_data = var_data.reshape(-1, ny, nx)
    else:
        var_data = var_data[np.newaxis, :, :]

    # Compute 2D FFT
    fft_data = np.fft.fft2(var_data, axes=(-2, -1))
    power_spectrum = (np.abs(fft_data) ** 2).mean(axis=0)

    # Rest of the function remains the same
    # Get wavenumbers
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)

    # Create 2D wavenumber grid
    kxx, kyy = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kxx**2 + kyy**2)

    # Create wavenumber bins for azimuthal averaging
    k_bins = np.logspace(
        np.log10(k_mag[k_mag > 0].min()), np.log10(k_mag.max()), num=50
    )

    # Perform azimuthal averaging
    k_averaged = []
    power_averaged = []

    for i in range(len(k_bins) - 1):
        k_mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if k_mask.any():
            k_averaged.append(np.mean(k_mag[k_mask]))
            power_averaged.append(np.mean(power_spectrum[k_mask]))

    # Convert to arrays
    k_averaged = np.array(k_averaged)
    power_averaged = np.array(power_averaged)

    # Calculate effective resolution
    effective_resolution = 1 / (4 * dx)

    # Remove first and last two wavenumbers
    return k_averaged[2:-2], power_averaged[2:-2], effective_resolution


def calculate_all_spectra(ds_gt, ds_ml, ds_nwp, variables):
    """Calculate and cache all energy spectra, including temporal evolution.

    Parameters
    ----------
    ds_gt : xarray.Dataset
        Ground truth dataset
    ds_ml : xarray.Dataset
        Machine learning model dataset
    ds_nwp : xarray.Dataset
        Numerical weather prediction model dataset
    variables : list
        List of variables to process

    Returns
    -------
    dict
        Nested dictionary containing wavenumbers, spectra and effective resolution
        for each dataset and variable, including temporal evolution
    """
    spectra_cache = {
        "gt": {},
        "ml": {"static": {}, "temporal": {}},
        "nwp": {"static": {}, "temporal": {}} if ds_nwp is not None else None,
    }

    for var in variables:
        if var not in ds_gt or var not in ds_ml:
            continue
        print(f"Calculating spectra for {var}...")

        # Calculate static spectra for ground truth
        k_gt, spec_gt, eff_res = calculate_energy_spectra(ds_gt[var])
        spectra_cache["gt"][var] = (k_gt, spec_gt, eff_res)

        # Calculate temporal spectra for ML model
        forecast_times = ds_ml.elapsed_forecast_duration.values
        ml_temporal = {}
        for time in forecast_times:
            data = ds_ml[var].sel(elapsed_forecast_duration=time)
            k, spec, _ = calculate_energy_spectra(data)
            ml_temporal[time] = (k, spec)

        spectra_cache["ml"]["temporal"][var] = ml_temporal
        spectra_cache["ml"]["static"][var] = calculate_energy_spectra(
            ds_ml[var]
        )

        # Calculate temporal spectra for NWP model if available
        if ds_nwp is not None and var in ds_nwp:
            forecast_times_nwp = ds_nwp.elapsed_forecast_duration.values
            nwp_temporal = {}
            for time in forecast_times_nwp:
                data = ds_nwp[var].sel(elapsed_forecast_duration=time)
                k, spec, _ = calculate_energy_spectra(data)
                nwp_temporal[time] = (k, spec)

            spectra_cache["nwp"]["temporal"][var] = nwp_temporal
            spectra_cache["nwp"]["static"][var] = calculate_energy_spectra(
                ds_nwp[var]
            )

    return spectra_cache

def plot_energy_spectra(spectra_cache, var, level=None, show_legend=False, temporal=False, ax=None, label=None, add_gt=True, add_eff_res=True, add_lsd=True):
    """Plot energy spectra comparison using pre-calculated spectra.

    Parameters
    ----------
    spectra_cache : dict
        Cache containing pre-calculated spectra
    var : str
        Variable name to plot
    level : float, optional
        Vertical level in hPa
    show_legend : bool, optional
        Whether to show the legend (default: False)
    temporal : bool, optional
        Whether to plot temporal evolution (default: False)
    label : str, optional
        Label for the plot (default: None)
    add_gt : bool, optional
        Whether to add ground truth to the plot (default: True)
    add_eff_res : bool, optional
        Whether to add effective resolution to the plot (default: True)
    add_lsd : bool, optional
        Whether to add LSD metric to the plot (default: True)
    """
    k_gt, spec_gt, eff_res = spectra_cache["gt"][var]
    k_ml, spec_ml, _ = spectra_cache["ml"]["static"][var]


    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)
    else:
        fig = ax.figure

    # Plot ground truth spectrum
    if add_gt:
        ax.loglog(
            k_gt,
            spec_gt,
            color=COLORS["gt"],
            label="ground truth",
            linestyle=LINE_STYLES["gt"][0],
            marker=LINE_STYLES["gt"][1],
            markevery=5,
        )

    if temporal:
        # Plot temporal evolution
        forecast_times = list(spectra_cache["ml"]["temporal"][var].keys())
        if label is None:
            label = f"ML-LES prediction (t={int(time / np.timedelta64(1, 'm'))} min)"
        for time in forecast_times:
            k_ml_t, spec_ml_t = spectra_cache["ml"]["temporal"][var][time]
            ax.loglog(
                k_ml_t,
                spec_ml_t,
                label=label,
                linestyle=LINE_STYLES["ml_tmp"][0],
                marker=LINE_STYLES["ml_tmp"][1],
                markevery=4,
                alpha=0.5
            )

    # Plot NWP spectrum if available
    if spectra_cache["nwp"] and var in spectra_cache["nwp"]["static"]:
        k_nwp, spec_nwp, _ = spectra_cache["nwp"]["static"][var]
        ax.loglog(
            k_nwp,
            spec_nwp,
            color=COLORS["nwp"],
            label="NWP",
            linestyle=LINE_STYLES["nwp"][0],
            marker=LINE_STYLES["nwp"][1],
            markevery=3,
        )

    # # Plot ML spectrum
    # ax.loglog(
    #     k_ml,
    #     spec_ml,
    #     color=COLORS["ml"],
    #     label="ML Model Prediction (Avg)",
    #     linestyle=LINE_STYLES["ml"][0],
    #     marker=LINE_STYLES["ml"][1],
    #     markevery=4,
    # )

    # Plot effective resolution
    if add_eff_res:
        ax.axvline(
            eff_res,
            color="salmon",
            linestyle="--",
            label="eff. model res.",
        )

    # Add LSD metric
    spec_nwp = (
        spectra_cache["nwp"]["static"][var][1]
        if spectra_cache["nwp"] and var in spectra_cache["nwp"]["static"]
        else None
    )
    if add_lsd:
        add_lsd_to_plot(ax, spec_gt, spec_ml, spec_nwp)

    # Customize plot
    ax.set_xlabel("wavenumber / m$^{-1}$")
    unit = VARIABLE_UNITS.get(var, "")
    ax.set_ylabel(f"energy density / {unit}$^{2}\cdot$ m")
    title = f"energy spectra comparison for {var}"
    if level is not None:
        title += f" at level {level} hPa"
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.5)

    return fig, ax


# def display_lsd_table(spectra_cache, variables, name, caption=""):
#     """Display and export LSD metrics table using pre-calculated spectra.

#     Parameters
#     ----------
#     spectra_cache : dict
#         Cache containing pre-calculated spectra
#     variables : list
#         List of variables to analyze
#     name : str
#         Name for exported files
#     caption : str, optional
#         Caption for LaTeX table
#     """
#     lsd_data = {var: {"ML": None, "NWP": None} for var in variables}

#     for var in variables:
#         if (
#             var not in spectra_cache["gt"]
#             or var not in spectra_cache["ml"]["static"]
#         ):
#             continue

#         spec_gt = spectra_cache["gt"][var][1]
#         spec_ml = spectra_cache["ml"]["static"][var][1]
#         spec_nwp = (
#             spectra_cache["nwp"]["static"][var][1]
#             if spectra_cache["nwp"] and var in spectra_cache["nwp"]["static"]
#             else None
#         )

#         lsd_ml, lsd_nwp = calculate_log_spectral_distance(
#             spec_gt, spec_ml, spec_nwp
#         )
#         lsd_data[var]["ML"] = lsd_ml
#         if lsd_nwp is not None:
#             lsd_data[var]["NWP"] = lsd_nwp

#     df = pd.DataFrame(lsd_data).T

#     # Display styled table
#     styled_df = df.style.format(
#         lambda x: f"{x:.3f}" if pd.notnull(x) else "-"
#     ).map(
#         lambda x: f"color: {'green' if x < 1 else 'orange' if x < 2 else 'red'}"
#         if pd.notnull(x)
#         else ""
#     )
#     display(styled_df)

#     # Export raw dataframe
#     export_table(df, name, caption)


def plot_wavenumber_evolution(
    spectra_cache,
    ds_ml,
    ds_nwp,
    variables,
    wavenumbers=[1e-2, 5e-2, 1e-1],
    level=None,
):
    """Plot wavenumber evolution using pre-calculated temporal spectra.

    Parameters
    ----------
    spectra_cache : dict
        Cache containing pre-calculated spectra including temporal evolution
    ds_ml : xarray.Dataset
        Machine learning model dataset (used only for time coordinates)
    ds_nwp : xarray.Dataset
        Numerical weather prediction model dataset
    variables : list
        List of variables to analyze
    wavenumbers : list, optional
        List of wavenumbers to analyze
    level : float, optional
        Vertical level in hPa
    """
    forecast_times = ds_ml.elapsed_forecast_duration.values

    for var in variables:
        if (
            var not in spectra_cache["gt"]
            or var not in spectra_cache["ml"]["temporal"]
        ):
            continue

        fig, axes = plt.subplots(1, len(wavenumbers), figsize=(24, 8), dpi=DPI)

        for idx, target_k in enumerate(wavenumbers):
            spectra = {
                "gt": [],
                "ml": [],
                "nwp": [] if spectra_cache["nwp"] else [],
            }

            # Get ground truth power for target wavenumber
            k_gt, spec_gt, _ = spectra_cache["gt"][var]
            idx_k_gt = np.abs(k_gt - target_k).argmin()
            gt_power = spec_gt[idx_k_gt]

            for time in forecast_times:
                # Get ML spectra from cache
                k_ml, spec_ml = spectra_cache["ml"]["temporal"][var][time]
                idx_k_ml = np.abs(k_ml - target_k).argmin()
                spectra["ml"].append(spec_ml[idx_k_ml])

                # Get NWP spectra from cache if available
                if (
                    spectra_cache["nwp"]
                    and var in spectra_cache["nwp"]["temporal"]
                    and time in spectra_cache["nwp"]["temporal"][var]
                ):
                    k_nwp, spec_nwp = spectra_cache["nwp"]["temporal"][var][
                        time
                    ]
                    idx_k_nwp = np.abs(k_nwp - target_k).argmin()
                    spectra["nwp"].append(spec_nwp[idx_k_nwp])
                else:
                    spectra["nwp"].append(np.nan)


                spectra["gt"].append(gt_power)

            # Plot evolution for current wavenumber
            hours = forecast_times / np.timedelta64(1, "h")

            # Export table
            # export_table(
            #     pd.DataFrame(
            #         {
            #             key.upper(): values
            #             for key, values in spectra.items()
            #             if values
            #         },
            #         index=hours,
            #     ).rename_axis(
            #         index="Elapsed Forecast Duration (h)",
            #         columns=f"Energy Density (k={target_k:.2e})",
            #     ),
            #     f"wavenumber_evolution_{var}_k{target_k:.2e}",
            # )

            # Plot spectra
            for key, values in spectra.items():
                if not values:
                    continue
                
                if not np.isnan(values).all():
                    axes[idx].plot(
                        hours,
                        values,
                        color=COLORS[key],
                        linestyle=LINE_STYLES[key][0],
                        marker=LINE_STYLES[key][1],
                        label=key.upper(),
                    )

            # Add titles and labels
            axes[idx].set_title(f"k = {target_k:.2f}")
            if idx == len(wavenumbers) // 2:  # Only middle plot gets x-label
                axes[idx].set_xlabel("Elapsed Forecast Duration (h)")
            if idx == 0:
                axes[idx].set_ylabel("Energy Density")
                axes[idx].legend()

        plt.tight_layout()
        plt.show()


def calculate_log_spectral_distance(true_spectrum, ml_spectrum, nwp_spectrum):
    """
    Calculate the Log Spectral Distance between three power spectra
    """
    eps = 1e-10
    log_spec1 = np.log10(true_spectrum + eps)
    log_spec2 = np.log10(ml_spectrum + eps)
    lsd_ml = np.sqrt(np.mean((log_spec1 - log_spec2) ** 2))
    if nwp_spectrum is None:
        return lsd_ml, None
    log_spec3 = np.log10(nwp_spectrum + eps)
    lsd_nwp = np.sqrt(np.mean((log_spec1 - log_spec3) ** 2))
    return lsd_ml, lsd_nwp


def add_lsd_to_plot(ax, true_spectrum, ml_spectrum, nwp_spectrum):
    """
    Add LSD metric as text box to spectrum plot
    """
    if nwp_spectrum is None:
        lsd_nwp = None
        lsd_ml, _ = calculate_log_spectral_distance(
            true_spectrum, ml_spectrum, None
        )
        textstr = f"LSD ML = {lsd_ml:.4f}"
    else:
        lsd_ml, lsd_nwp = calculate_log_spectral_distance(
            true_spectrum, ml_spectrum, nwp_spectrum
        )
        textstr = f"LSD NWP = {lsd_nwp:.4f}, LSD ML = {lsd_ml:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.1,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )