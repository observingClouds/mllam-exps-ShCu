"""
Wrapper around neural-lam.train to allow passing parameters via yaml file, e.g. params.yaml
"""

import argparse
import os
import sys
import yaml

import dvc.api
from loguru import logger

from neural_lam.train_model import main as train_model


def main(params):
    train_model(params)

def none_or_str(value):
    if value == 'None':
        return None
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="params.yaml")
    parser.add_argument("--eval", type=none_or_str, default=None, choices=[None, "val", "test"])
    parser.add_argument("--load", type=none_or_str, default=None, help="Path to model checkpoint to load if evaluating")
    parser.add_argument("--logger", type=str, default="mlflow")
    options = parser.parse_args()
    if options.eval is not None:
        params = dvc.api.params_show(stages=["evaluate"])
    else:
        params = dvc.api.params_show(stages=["train"])
    params.update(vars(options))
    print(params)

    if options.eval is not None and options.load is None:
        raise ValueError("Must provide --load when evaluating")
    args = []
    for k, v in params.items():
        args.extend([f"--{k}", str(v)])
    # args = params
    logger.info(f"Initiating run with parameters: {args}")
    main(args)
