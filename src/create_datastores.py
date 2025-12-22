"""
Create mllam-data-prep datastores for each subdomain
"""
import argparse
import matplotlib.pylab as plt
import numpy as np
import tqdm
import xarray as xr
import yaml

import sys
sys.path.append("./data")
from domains import domains


## Defining local projection
def set_projection(lats, lons):
    lat_0 = np.mean(lats)
    lon_0 = np.mean(lons)
    projection = {
        'class_name': 'LambertConformal',
        'kwargs': {
            'central_longitude': float(lon_0),
            'central_latitude': float(lat_0),
            'standard_parallels': [float(lat_0), float(lat_0)],
            'globe': {
                'semimajor_axis': 6367470.0,
                'semiminor_axis': 6367470.0,
            },
        }
    }
    return projection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create datastores for each domain")
    parser.add_argument('--domains', nargs='*', default=[], help='Domain names to process')
    parser.add_argument('--boundary-input-variables', nargs='+', default=None)
    parser.add_argument('--interior-input-variables', nargs='+', default=None)
    args = parser.parse_args()
    domains = {k: domains[k] for k in args.domains}

    ds = xr.open_zarr("/home/has/data/sources/icon/ICON312_v1.0.0.zarr", consolidated=True)

    boundary_input_variables = args.boundary_input_variables
    interior_input_variables = args.interior_input_variables

    ## Load existing datastore
    with open("./data/template/datastore.interior.yaml") as f:
        datastore = yaml.safe_load(f)

    with open("./data/template/datastore.boundary.yaml") as f:
        datastore_boundary = yaml.safe_load(f)

    with open("./data/template/config.yaml") as f:
        config = yaml.safe_load(f)

    # %%
    ## Create new datastores for each domain
    for domain_name, domain in tqdm.tqdm(domains.items(), desc="Creating datastores for domain"):
        cells = np.hstack([np.arange(s.start, s.stop) for s in domain["cells"]]).tolist()
        lats = np.rad2deg(ds.isel(cell=cells)['clat'])
        lons = np.rad2deg(ds.isel(cell=cells)['clon'])
        projection = set_projection(lats, lons)
        print(f"Number of cells: {len(cells)}")
        datastore_domain = datastore.copy()
        datastore_domain['inputs']['icon_merged']['coord_ranges']['cell'] = None
        datastore_domain['inputs']['icon_merged']['variables'] = interior_input_variables
        datastore_domain['inputs']['forcing_features']['coord_ranges']['cell'] = cells
        datastore_domain['inputs']['static_features']['coord_ranges']['cell'] = None
        datastore_domain['extra'] = {'projection': projection}
        string = yaml.dump(datastore_domain)
        string = string.replace("cell: null", "cell: *id001")
        string = string.replace("cell:\n      -", "cell: &id001\n      -")
        with open(f"./data/experiment/datastore.interior.{domain_name}.yaml", "w") as f:
            f.write(string)
        
        with open(f"./data/experiment/datastore.boundary.{domain_name}.yaml", "w") as f:
            datastore_boundary_domain = datastore_boundary.copy()
            datastore_boundary_domain['inputs']['icon_merged']['variables'] = boundary_input_variables
            datastore_boundary_domain['extra'] = {'projection': projection}
            datastore_boundary_domain["output"]["domain_cropping"]['interior_dataset_config_path'] = f"/home/has/repos/mllam-exps-ShCu/data/experiment/datastore.interior.{domain_name}.yaml"
            yaml.dump(datastore_boundary_domain, f)
        
        with open(f"./data/experiment/config.{domain_name}.yaml", "w") as f:
            config_domain = config.copy()
            config_domain['datastore']['config_path'] = f"/home/has/repos/mllam-exps-ShCu/data/experiment/datastore.interior.{domain_name}.yaml"
            config_domain['datastore_boundary']['config_path'] = f"/home/has/repos/mllam-exps-ShCu/data/experiment/datastore.boundary.{domain_name}.yaml"
            yaml.dump(config_domain, f)