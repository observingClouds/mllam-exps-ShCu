import os
import pickle
import types

import numpy as np


def grid_selection(
    grid,
    lats=types.MappingProxyType({"lats": [11, 15]}),
    lons=types.MappingProxyType({"lons": [-59.3, -55.3]}),
):
    # Subsection
    x_range = lons
    y_range = lats

    # Create grid-mask
    cell = (
        (grid.face_lat.values >= y_range[0])
        & (grid.face_lat.values <= y_range[1])
        & (grid.face_lon.values >= x_range[0])
        & (grid.face_lon.values <= x_range[1])
    )

    return cell


def load_grid_subset(dom, lats, lons, grid, path=".", return_grid=False):
    pkl_filename = os.path.join(
        path,
        f"cells_DOM0{dom}_lats{'-'.join(map(str,lats))}_lons{'-'.join(map(str,lons))}.pkl",
    )
    if not os.path.exists(pkl_filename) or return_grid:
        print("Creating cell mask")
        cell_subsection, grid_subsection = grid_selection(grid, lats, lons)
        with open(pkl_filename, "wb") as f:
            pickle.dump(cell_subsection, f)
    else:
        print("Reading cell mask")
        with open(pkl_filename, "rb") as f:
            cell_subsection = pickle.load(f)
    if return_grid:
        return cell_subsection, grid_subsection
    else:
        return cell_subsection
    
# _, g = load_grid_subset(2, [10,12],[-59,-57], ds_grid, return_grid=True)