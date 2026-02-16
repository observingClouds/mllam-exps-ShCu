"""
The surface values and radiation values have an offset of 10min.
This script corrects the offset by shifting the lookups for the kerchunk files.
"""

import json
import tqdm
import xarray as xr

ref_filename = "ablation_study/index.boundary.lw.json"
forcing_features_to_shift = ["thb_t", "sob_t"]
state_features_to_shift = ["thb_t", "sob_t"]
shift_by_time_chunks = -1

ds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={"consolidated": False}, storage_options={"fo": ref_filename})

with open(ref_filename, 'r') as f:
    refs_dict = json.load(f)

def get_feature_chunk_number(input_type, feature_name):
    """
    Get the chunk number for a given feature name in the dataset.
    """
    features = ds[f"{input_type}_feature"].values
    for i, feature in enumerate(features):
        if feature_name == feature:
            return i
    print(f"Feature {feature_name} not found")

def get_feature_chunks(input_type, feature_name):
    """
    Get all chunks for the given feature name.
    
    E.g. get all time and gridcells chunks for the feature
    like 0.0.0, 0.1.0, 0.2.0, ...
    """
    chunk_number = get_feature_chunk_number(input_type, feature_name)
    feature_chunks = []
    refs = refs_dict.get("refs", {})
    prefix = f"{input_type}/{chunk_number}."
    feature_chunks = [key for key in refs.keys() if key.startswith(prefix)]
    return feature_chunks

def shift_feature_in_df(input_type, feature_name, shift_by):
    """
    Shift the chunk number for a given feature in the refs dictionary.
    """
    chunks = get_feature_chunks(input_type, feature_name)
    refs = refs_dict.get("refs", {})
    new_refs = {}
    chunks_to_drop = []
    
    for chunk in tqdm.tqdm(chunks):
        parts = chunk.split("/")[1].split(".")
        time_chunk_number = int(parts[1])
        new_time_chunk_number = time_chunk_number + shift_by
        
        if new_time_chunk_number < 0:
            continue  # Skip shifting if it goes negative
        
        new_chunk = f"{input_type}/{parts[0]}.{new_time_chunk_number}.{parts[2]}"
        new_refs[new_chunk] = refs[chunk]
    
    # Handle deletion of last chunk when shifting forward
    if shift_by > 0 and chunks:
        sorted_chunks = sorted(chunks, key=lambda x: int(x.split(".")[1]))
        last_chunk = sorted_chunks[-1]
        parts = last_chunk.split(".")
        new_time_chunk_number = int(parts[1]) + shift_by
        new_chunk = f"{input_type}/{parts[0]}.{new_time_chunk_number}.{parts[2]}"
        chunks_to_drop.append(new_chunk)
    
    # Remove old chunks and add new ones
    for chunk in chunks:
        refs.pop(chunk, None)
    refs.update(new_refs)
    
    # Drop chunks that go beyond bounds
    for chunk in chunks_to_drop:
        refs.pop(chunk, None)

if "boundary" in ref_filename:
    feature_types = ["forcing"]
elif "interior" in ref_filename:
    feature_types = ["forcing", "state"]

for input_type in ["forcing"]:
    features_to_shift = forcing_features_to_shift if input_type == "forcing" else state_features_to_shift
    for feature_name in features_to_shift:
        shift_feature_in_df(input_type, feature_name, shift_by_time_chunks)

with open("ablation_study/index.boundary.lw.corrected.json", 'w') as f:
    json.dump(refs_dict, f)
