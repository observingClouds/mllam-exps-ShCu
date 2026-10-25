# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "virtualizarr @ git+https://github.com/observingClouds/VirtualiZarr.git@mllam-exps-ShCu",
#   "loguru",
#   "ipdb",
#   "zarr==3.1.6",
# ]
# ///
"""
Create a subset of the mllam-data-prep dataset by using kerchunk references.

This script creates a virtual subset of the mllam-data-prep for e.g. ablation
studies without duplicating data.

Call the script like:
    uv run --script src/create_virtual_subset.py \
        --input /home/has/repos/mllam-exps-ShCu/data/experiment/data/datastore.boundary.domain03.zarr \
        --output index.boundary.json --subset-type forcing_feature \
        --variables qv_2m tqc_dia t_2m

See also: https://github.com/mllam/mllam-data-prep/issues/88
"""
from loguru import logger
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import ZarrParser, KerchunkJSONParser
from obstore.store import LocalStore, from_url
from virtualizarr.registry import ObjectStoreRegistry
from pathlib import Path
import argparse
import numpy as np
import xarray as xr

logger.info("Creating virtual subset of mllam-data-prep dataset")

parser = argparse.ArgumentParser(description="Create virtual subset from zarr store")
parser.add_argument('--input', '-i', dest='zarr_store',
                    help="Path to the mllam-data-prep input zarr store")
parser.add_argument('--output', '-o', dest='reference', default='index.json',
                    help='Path to output kerchunk index')
parser.add_argument('--subset-type', '-s', dest='subset_type', default='state_feature',
                    help='Type of subset to create: state_feature or forcing_feature')
parser.add_argument('--variables', '-v', dest='variables', nargs='+', default=None,
                    help='List of variables to include in the subset. By default include all.')
parser.add_argument('--replacement-variable', dest='replacement_variable', default=None,
                    help='Replacement variable name to inject into the subset.')
parser.add_argument('--replacement-zarr-store', dest='replacement_zarr_store',
                    help='Path to replacement zarr store.')
args = parser.parse_args()


if args.zarr_store.endswith('.json'):
    zarr_store = str(args.zarr_store)
    file_url = f"file://{zarr_store}"
    store = from_url("file://")
    registry = ObjectStoreRegistry({"file://": store})
    registry.register(file_url, store)
    parser = KerchunkJSONParser()
    vds = open_virtual_dataset(url=zarr_store,registry=registry,parser=parser)
else:
    zarr_store = str(args.zarr_store)
    store = LocalStore(prefix=zarr_store)
    registry = ObjectStoreRegistry({f"file://{zarr_store}": store})
    parser = ZarrParser()
    vds = open_virtual_dataset(url=zarr_store,registry=registry,parser=parser)

# Select subset of mllam-data-prep dataset, e.g. reduce number of state_features
# Exclude replacement variable from original data if specified
if args.variables is not None:
    variables_to_use = [v for v in args.variables if v != args.replacement_variable] if args.replacement_variable else args.variables
    slices = list(slice(int(x),int(x+1)) for x in np.argwhere(np.isin(vds[args.subset_type],variables_to_use))[:,0])
    original_data_slices = [vds.isel({args.subset_type: s}) for s in slices]
else:
    # If no variables specified but replacement_variable is, exclude it from all variables
    if args.replacement_variable is not None:
        all_vars = vds[args.subset_type].values
        variables_to_use = [v for v in all_vars if v != args.replacement_variable]
        slices = list(slice(int(x),int(x+1)) for x in np.argwhere(np.isin(vds[args.subset_type],variables_to_use))[:,0])
        original_data_slices = [vds.isel({args.subset_type: s}) for s in slices]
    else:
        original_data_slices = [vds]

# Handle replacement variable if provided
if args.replacement_variable is not None and args.replacement_zarr_store is not None:
    replacement_variables = args.replacement_variable
    replacement_zarr_store = args.replacement_zarr_store
    replacement_store = LocalStore(prefix=replacement_zarr_store)
    replacement_registry = ObjectStoreRegistry({f"file://{replacement_zarr_store}": replacement_store})
    replacement_parser = ZarrParser()
    replacement_vds = open_virtual_dataset(url=replacement_zarr_store,registry=replacement_registry,parser=replacement_parser)
    
    replacement_data_slices = [replacement_vds.isel({args.subset_type: slice(0,1)})]
    joint_slices = original_data_slices + replacement_data_slices
    
    try:
        vds = xr.concat(joint_slices, dim=f'{args.subset_type}', data_vars='minimal', coords='minimal')
    except Exception:
        vds = xr.concat([*[original_data_slices[i][['state','state__train__std','static','static__train__std','static__train__mean','splits','forcing',
                                    'forcing__train__diff_std','forcing__train__mean','forcing__train__std','forcing__train__diff_mean',
                                    'state__train__mean', 'state__train__diff_mean', 'state__train__diff_std']] for i in range(len(original_data_slices))], 
                                    joint_slices[-1][['state','state__train__std','state__train__mean', 'state__train__diff_mean', 
                                                      'state__train__diff_std']]],
                                                      dim=f'{args.subset_type}', data_vars='minimal', coords='minimal')
else:
    # No replacement - just concat original data slices if needed
    if len(original_data_slices) > 1:
        vds = xr.concat(original_data_slices, dim=f'{args.subset_type}', data_vars='minimal', coords='minimal')
    elif len(original_data_slices) == 1:
        vds = original_data_slices[0]
    # else vds remains as originally loaded

vds.vz.to_kerchunk(args.reference, format='json')

# New subsetted mllam-data-prep dataset just containing references to original mllam-data-prep (e.g. no extra copy)
logger.info("Virtual subset created. You can open it with: `xr.open_zarr('reference://', storage_options={'fo': args.reference})`")