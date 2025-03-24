import xarray as xr
import numpy as np

def icon_grid_2_ugrid(xr_grid: xr.Dataset) -> xr.Dataset:
    """
    Convert an ICON grid to being UGRID-compatible.

    Parameters:
    icon_grid_fname (str): The file path of the ICON grid dataset.

    Returns:
    ugrid_fname (str): The file path of the UGRID-compatible grid dataset.

    References:
    - UGRID conventions:
    http://ugrid-conventions.github.io/ugrid-conventions
    """

    # Adapt from Fortran zero basednes... only for real index fields (not all
    # int32 arrays contain indices) for example refin_ctl whereas in ICON
    # indices and refin_ctl values
    index_lists = [
        "cell_index",
        "edge_index",
        "vertex_index",
        "parent_cell_index",
        "parent_edge_index",
        "parent_vertex_index",
        "start_idx_c",
        "end_idx_c",
        "start_idx_e",
        "end_idx_e",
        "start_idx_v",
        "end_idx_v",
        "neighbor_cell_index",
        "edge_of_cell",
        "vertex_of_cell",
        "adjacent_cell_of_edge",
        "edge_vertices",
        "cells_of_vertex",
        "edges_of_vertex",
        "vertices_of_vertex",
    ]

    for vname in index_lists:
        if vname in set(xr_grid.data_vars.keys()):
            #
            # make it zero-indexed
            xr_grid[vname].data = np.where(
                xr_grid[vname].data > 0, xr_grid[vname].data - 1, -1
            )
            xr_grid[vname].attrs["start_index"] = 0
            xr_grid[vname].attrs["_FillValue"] = -1
            #
            # if needed, transpose
            vshape = xr_grid[vname].shape
            if len(vshape) == 2 and (vshape[0] < vshape[1]):
                xr_grid[vname] = xr.DataArray(
                    data=xr_grid[vname].data.T,
                    dims=xr_grid[vname].dims[::-1],
                    coords=xr_grid[vname].coords,
                    attrs=xr_grid[vname].attrs,
                )

    # Store topology information
    xr_grid["mesh"] = xr.DataArray(
        -1,  # Dummy value for creating the DataArray with attributes
        attrs=dict(
            cf_role="mesh_topology",
            topology_dimension=2,
            node_dimension="vertex",
            edge_dimension="edge",
            face_dimension="cell",
            node_coordinates="vlon vlat",
            edge_coordinates="elon elat",
            face_coordinates="clon clat",
            face_node_connectivity="vertex_of_cell",
            edge_node_connectivity="edge_vertices",
            face_edge_connectivity="edge_of_cell",
            face_face_connectivity="neighbor_cell_index",
            edge_face_connectivity="adjacent_cell_of_edge",
        ),
    )


    return xr_grid