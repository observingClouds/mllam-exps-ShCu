"""Custom MDP functions."""

import numpy as np
from mllam_data_prep.ops.derive_variable.physical_field import calculate_toa_radiation

def compute_toa_radiation_radians(lat, lon, time):
    """Compute TOA radiation in radians."""
    return calculate_toa_radiation(np.rad2deg(lat), np.rad2deg(lon), time)