# AGENTS instructions

## Local development environment
- Use `uv run python` to test and run any individual script snippets.

## Structure of the mllam-exps-ShCu repository
This repository orchestrates limited area machine learning experiments in the subtropics with a focus on shallow convection. The structure is as follows:

- `dvc.yaml`: DVC pipeline definition file that specifies the stages of the experiment workflow, including data processing, model training, and evaluation (some parts might be missing)
- `mllam_data_prep`: Sub-repository containing tools to create training datasets from raw model output.
- `neural_lam`: Sub-repository containing the main code for training and evaluating the machine learning models.
- `src`, `analysis`: Directories for additional scripts and analysis scripts used mostly for postprocessing of the evaluation results.