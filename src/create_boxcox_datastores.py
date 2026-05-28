"""Create boxcox-transformed boundary and interior datastores.

This script extracts the implementation-test logic from the notebook and makes
the transform factor and target variable configurable from the command line.

Example:
    uv run python src/create_boxcox_datastores.py \
        --factor 0.06666666666666667 \
        --variable tqc_dia
    uv run python src/create_boxcox_datastores.py \
        --factor -0.08907066350763708 \
        --variable rain_gsp_rate \
        --positive-only \
        --nonpositive-constant -80.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.special
import xarray as xr
import yaml


DEFAULT_BOUNDARY_INPUT = Path("./data/experiment/data/datastore.boundary.domain03.zarr")
DEFAULT_INTERIOR_INPUT = Path("./data/experiment/data/datastore.interior.domain03.zarr")


def build_output_path(
    input_path: Path,
    variable: str,
    factor: float,
    *,
    positive_only: bool,
) -> Path:
    factor_tag = str(factor).replace("-", "neg").replace(".", "p")
    mode_tag = "posonly" if positive_only else "all"
    return input_path.with_name(f"{input_path.stem}.boxcox{variable}_{mode_tag}_{factor_tag}.zarr")


def load_train_split(dataset: xr.Dataset) -> dict[str, str]:
    config = yaml.safe_load(dataset.attrs["creation_config"])
    return config["output"]["splitting"]["splits"]["train"]


def build_encoding(source_dataset: xr.Dataset, transformed_dataset: xr.Dataset, field_name: str) -> dict[str, dict[str, object]]:
    field_compressor = source_dataset[field_name].encoding.get("compressor")
    encoding: dict[str, dict[str, object]] = {}
    for var in transformed_dataset.data_vars:
        encoding[var] = {"compressor": field_compressor}
    return encoding


def transform_dataset(
    dataset: xr.Dataset,
    *,
    feature_dim: str,
    field_name: str,
    variable: str,
    factor: float,
    positive_only: bool,
    nonpositive_constant: float,
    output_path: Path,
) -> None:
    selected = dataset.sel({feature_dim: [variable]})
    if positive_only:
        positive_mask = selected[field_name] > 0
        positive_values = selected[field_name].where(positive_mask, other=1.0)
        if np.isclose(factor, 0.0):
            transformed_field = np.log(positive_values)
        else:
            transformed_field = (positive_values**factor - 1) / factor
        selected[field_name] = xr.where(positive_mask, transformed_field, nonpositive_constant)
    else:
        selected[field_name] = scipy.special.boxcox(selected[field_name], factor)

    train_split = load_train_split(selected)
    selected_train = selected.sel(
        {"time": slice(train_split["start"], train_split["end"])}
    )

    selected[f"{field_name}__train__diff_mean"] = selected_train[field_name].diff(dim="train").mean(dim=["time", "grid_index"])
    selected[f"{field_name}__train__diff_std"] = selected_train[field_name].diff(dim="train").std(dim=["time", "grid_index"])
    selected[f"{field_name}__train__mean"] = selected_train[field_name].mean(dim=["time", "grid_index"])
    selected[f"{field_name}__train__std"] = selected_train[field_name].std(dim=["time", "grid_index"])

    drop_vars = [
        "clat",
        "clon",
        "static_feature_long_name",
        "static_feature_source_dataset",
        "static_feature_units",
        "static",
        "splits",
    ]
    if feature_dim == "state_feature":
        drop_vars.extend([
            "forcing",
            "forcing__train__diff_mean",
            "forcing__train__diff_std",
            "forcing__train__mean",
            "forcing__train__std",
        ])
    else:
        drop_vars.extend([
            "static__train__mean",
            "static__train__std",
        ])

    selected_out = selected.drop_vars([var for var in drop_vars if var in selected])
    encoding = build_encoding(dataset, selected_out, field_name)
    selected_out.to_zarr(output_path, mode="w", encoding=encoding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create boxcox-transformed boundary and interior datastores."
    )
    parser.add_argument(
        "--factor",
        type=float,
        required=True,
        help="Box-Cox factor to apply to the selected variable.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Transform only positive values and fill nonpositive values with a constant.",
    )
    parser.add_argument(
        "--nonpositive-constant",
        type=float,
        default=-80.0,
        help="Value used for entries that are not transformed when --positive-only is set.",
    )
    parser.add_argument(
        "--variable",
        required=True,
        help="Feature name to transform, for example tqc_dia or rain_gsp_rate.",
    )
    parser.add_argument(
        "--boundary-input",
        type=Path,
        default=DEFAULT_BOUNDARY_INPUT,
        help="Path to the boundary datastore.",
    )
    parser.add_argument(
        "--interior-input",
        type=Path,
        default=DEFAULT_INTERIOR_INPUT,
        help="Path to the interior datastore.",
    )
    parser.add_argument(
        "--boundary-output",
        type=Path,
        default=None,
        help="Path to the transformed boundary datastore.",
    )
    parser.add_argument(
        "--interior-output",
        type=Path,
        default=None,
        help="Path to the transformed interior datastore.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boundary_output = args.boundary_output or build_output_path(
        args.boundary_input,
        args.variable,
        args.factor,
        positive_only=args.positive_only,
    )
    interior_output = args.interior_output or build_output_path(
        args.interior_input,
        args.variable,
        args.factor,
        positive_only=args.positive_only,
    )

    ds_boundary = xr.open_zarr(args.boundary_input)
    ds_interior = xr.open_zarr(args.interior_input)

    transform_dataset(
        ds_boundary,
        feature_dim="forcing_feature",
        field_name="forcing",
        variable=args.variable,
        factor=args.factor,
        positive_only=args.positive_only,
        nonpositive_constant=args.nonpositive_constant,
        output_path=boundary_output,
    )

    transform_dataset(
        ds_interior,
        feature_dim="state_feature",
        field_name="state",
        variable=args.variable,
        factor=args.factor,
        positive_only=args.positive_only,
        nonpositive_constant=args.nonpositive_constant,
        output_path=interior_output,
    )


if __name__ == "__main__":
    main()