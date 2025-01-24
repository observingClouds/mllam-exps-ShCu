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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="params.yaml")
    options = parser.parse_args()
    params = dvc.api.params_show(stages=["train"])
    params.update(vars(options))
    args = []
    for k, v in params.items():
        args.extend([f"--{k}", str(v)])
    logger.info(f"Initiating run with parameters: {args}")
    main(args)
