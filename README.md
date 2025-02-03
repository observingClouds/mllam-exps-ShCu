# Orchestrating MLLAM Experiments with DVC

This repository contains a [DVC](dvc.org) pipeline to train, reproduce and share experiments for the Neural-LAM project.

## Design
Pipelines consist of several stages, each stage describing a single step in the experiment process. Main stages for an ML experiment consist
typically of data preparation, model training, evaluation and inference. These stages are defined in the `dvc.yaml` file. For the Neural-LAM
project, the main stages are _prepare dataset_, _create graph_, _train_, _evaluate_. The stages are linked by dependencies (`deps`) and outputs (`out`).
DVC automatically tracks the dependencies and outputs and only runs stages that have been updated. In the background DVC uses git for these purposes.
In addition, also paramters like model version, num_workers, epochs, ... need to be tracked. These are defined in `params.yaml` files, here
particularly in `data/training_params.yaml` and `data/evaluate_params.yaml`.

Generally, experiments can be run with `dvc exp run`, but since we need to schedule our experiments with SLURM and DVC relies on being called
after a job has been finished to calculate the checksums, we need to do some hacks:

1. Use `sbatch -W` in the `cmd` field of the stage to return to DVC only after the model task has been finished.
2. Use a tmux session to run DVC in the background and let it wait for the job to finish.
3. Use wrapper scripts to ingest SLURM environment variables and setups.

The overall structure of the repository is as follows:

```plaintext
mllam-exps
├── data
│   ├── datastore.zarr  # output from the prepare dataset stage (not there initially)
│   ├── graph  # output from the create graph stage (not there initially)
│   ├── config.yaml  # configuration file for the Neural-LAM project
│   ├── datastore.yaml  # configuration file for the data preparation stage via mllam-data-prep
│   ├── training_params.yaml  # configuration file for the training stage
│   └── evaluate_params.yaml  # configuration file for the evaluation stage
├── logs  # log files from SLURM jobs
├── machines  # machine configurations for the Neural-LAM project
│   ├── environment.sh  # source environment incl. python modules
│   └── slurm.evaluate.sh  # SLURM wrapper script for evaluation
├── dvc.lock  # lock file for DVC containing checksums for the latest stages
├── dvc.yaml  # DVC pipeline definition
├── train_wrapper.py  # wrapper script to allow neural-lam parameters be passed via DVC/params.yaml
├── version.mllam.txt  # Lock file for neural-lam version (external dependencies are not well supported by DVC)
```

## Setup
To setup DVC, DVC needs to be installed:
```bash
pip install dvc --target packages_dvc
export PYTHONPATH=packages_dvc:$PYTHONPATH
```

### Sharing experiments and data
To share experiments including models and metrics, a common remote needs to be defined. DVC supports a long list of
remotes, but on a cluster system that potentially does not have access to the internet and e.g. S3 buckets,
a local remote can be used. To setup a local remote, run:

```bash
dvc remote add -d localshare /dcai/projects/cu_0003/dvc/
```

When the remote is created for the first time (i.e. by the first user creating this folder), it needs to be initialized as git repository
and the access rights configured for group write and read:

```bash
cd /dcai/projects/cu_0003/dvc/
chmod o+rwx .
git init
```

Further info can be found in the [DVC documentation](https://dvc.org/doc/user-guide/experiment-management/sharing-experiments).

## Running an experiment

Experiments are generally run by:

```bash
dvc exp run
```

> **Note**: It is highly advised to run this in a tmux session as the command will take a while to finish (`sbatch -W`).

DVC automatically runs only the stages that have been updated as it keeps track of the dependencies defined in dvc.yaml.

Once an experiment has been run and it is worth sharing with others, it can be pushed to the remote:

```bash
dvc exp push /dcai/projects/cu_0003/dvc/ <experiment>
```
This command will push all tracked experiment data (model, graph, data, ...) to the remote.

In the same way, experiments can be pulled from the remote:

```bash
dvc exp pull /dcai/projects/cu_0003/dvc/ <experiment>
```

And if one is just curious which experiments are available:

```bash
dvc exp list /dcai/projects/cu_0003/dvc/ # on the remote
dvc exp list # in the user/local space
```

### Running a range of experiments
For hypterparameter searches or testing different configurations, it is possible to run a range of experiments. This can be done
by using the `--queue` option:

```bash
dvc exp run --queue -S data/training_params.yaml:hidden_dim='1,2,4,8,16'
dvc queue start
```

This will run the training stage with different hidden dimensions. The `--queue` option will create a sequence of experiments that
are defined by changing the parameters on-the-fly defined in e.g. `data/training_params.yaml` with the `-S` option.

## Patches

Some tweaks are necessary to make DVC work with the Neural-LAM project.

### Deactive hyperparameter logging
The hyperparameter logging does not work as the NeuralLAMConfig objects cannot be serialized with ruamel.yaml by default.
Therefore the logging of hyperparameters has been removed from `.../site-packages/pytorch_lightning/trainer/trainer.py`:

```suggestion
- _log_hyperparams(self)
+ # _log_hyperparams(self)
```
