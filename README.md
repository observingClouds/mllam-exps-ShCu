# Orchestrating MLLAM Experiments with DVC

This repository contains [DVC](dvc.org) pipeline definition to train, reproduce and share experiments for the Neural-LAM project.

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

DVC automatically runs only the stages that have been updated as it keeps track of the dependencies defined in dvc.yaml.

Once an experiment has been run and worth sharing with others, it can be pushed to the remote:

```bash
dvc exp push <experiment> /dcai/projects/cu_0003/dvc/
```
This command will push all tracked experiment data (model, graph, data, ...) to the remote.

In the same way, experiments can be pulled from the remote:

```bash
dvc exp pull <experiment> /dcai/projects/cu_0003/dvc/
```

And if one is just curious which experiments are available:

```bash
dvc exp list /dcai/projects/cu_0003/dvc/ # on the remote
dvc exp list # in the user/local space
```

## Patches

Some tweaks are necessary to make DVC work with the Neural-LAM project.

### Deactive hyperparameter logging
The hyperparameter logging does not work as the NeuralLAMConfig objects cannot be serialized with ruamel.yaml by default.
Therefore the logging of hyperparameters has been removed from `.../site-packages/pytorch_lightning/trainer/trainer.py`:

```suggestion
- _log_hyperparams(self)
+ # _log_hyperparams(self)
```
