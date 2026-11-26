# Orchestrating MLLAM Experiments with DVC

This repository contains a [DVC](dvc.org) pipeline to train, reproduce and share experiments for the Neural-LAM project.

>[!Warning]
>This package is still under heavy development and currently has also a tendency to be geared towards a specific HPC environment.

## Initial transfer of input data
The data can be transfered from the cloud via:
```bash
git clone https://github.com/observingClouds/mllam-exps-ShCu.git
cd mllam-exps-ShCu
curl -LsSf https://astral.sh/uv/install.sh | sh   # see https://docs.astral.sh/uv/getting-started/installation/ for other options
uv run python src/preprocess.py --output /dcai/projects/cu_0003/data/sources/icon/ICON312_v1.0.0.zarr
```
This step will take about 1-2 hours depending on the network connection and used resources.

## Design
Pipelines consist of several stages, each stage describes a single step in the experiment process. Main stages for an ML experiment consist
typically of data preparation, model training, evaluation and inference. These stages are defined in the `dvc.yaml` file. For the Neural-LAM
project, the main stages are _prepare dataset_, _create graph_, _train_, _evaluate_. The stages are linked by dependencies (`deps`) and outputs (`out`).
DVC automatically tracks the dependencies and outputs and only runs stages that have been updated. In the background DVC uses git for these purposes.
In addition, also paramters like model version, num_workers, epochs, ... need to be tracked. These are defined in `params.yaml` files, here
particularly in `data/training_params.yaml` and `data/evaluate_params.yaml`.

Generally, experiments can be run with `dvc exp run`, but since we need to schedule our experiments with SLURM and DVC relies on being called
after a job has been finished to calculate the checksums, we need to do some hacks:

1. Use `sbatch -W` in the `cmd` field of the stage to return to DVC only after the model tasks has been finished.
2. Use a tmux session to run DVC in the background and let it wait for the job to finish.
3. Use wrapper scripts to ingest SLURM environment variables and setups.

The overall structure of the repository is as follows:

```plaintext
mllam-exps
├── data
│   ├── datastore.zarr  # output from the prepare dataset stage (not there initially)
│   ├── graph  # output from the create graph stage (not there initially)
│   ├── config.yaml  # configuration file for the Neural-LAM project
│   └── datastore.yaml  # configuration file for the data preparation stage via mllam-data-prep
├── logs  # log files from SLURM jobs
├── machines  # machine configurations for the Neural-LAM project
│   ├── environment.sh  # source environment incl. python modules
│   ├── check_for_venv_path.sh  # script to check if path to virtual python environment is set 
│   ├── slurm.neural-lam.sh  # SLURM wrapper script for training and evaluation
│   └── slurm.mllam-data-prep.sh  # SLURM wrapper script for mllam-data-prep
├── dvc.lock  # lock file for DVC containing checksums for the latest stages
├── dvc.yaml  # DVC pipeline definition
├── params.yaml  # Parameters to train and evaluate model, which are converted to command line arguments to neural_lam
├── version.mllam.txt  # Lock file for neural-lam version (external dependencies are not well supported by DVC)
```

## Setup
To setup DVC, DVC needs to be installed:
```bash
pip install dvc
```

To be able to track experiments with MLflow, add your username and password to a `~/.bashrc`:
```
#!/bin/bash
export MLFLOW_TRACKING_USERNAME=""
export MLFLOW_TRACKING_PASSWORD=""
#empty line
```

Install neural-lam and dependencies into a virtual environment, and set the
environment variable `MLLAM_VENV_PATH` to the path of the virtual environment, e.g.:
```bash
export MLLAM_VENV_PATH=/dcai/users/denlef/git-repos/mllam/mllam-exps/.venv
```
A good place for this is also the `~/.bashrc`

### Sharing a DVC cache
This repository is set up for a common DVC cache, which allows to automatically pull the results of stages and experiments
someone else has already run. E.g. in the case of the baseline dataset, only the first person had to create this dataset
for all other team members this dataset will be pulled automatically from cache.

For the cache to work properly, ensure that you are using a group that all members are using, so that there won't be
permission errors. A good way to ensure this, add the following to the `~/.bashrc`:

```bash
# `newgrp` launches a new shell, so to avoid an infinite loop, check if the group is already set
# Only run for interactive shells
if [[ $- == *i* ]]; then
    if [ "$(id -gn)" != "cu_0003" ]; then
        newgrp cu_0003
    fi
fi
```
where `cu_0003` is the group that all team members belong to.

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

#### Hetzner storage box (WebDAV)
The default remote `hetzner` is a Hetzner storage box accessed via WebDAV over HTTPS (outgoing SSH is blocked on
some networks). Store the password locally (not committed):

```bash
uv run dvc remote modify --local hetzner password '<storagebox-password>'
```

The server redirects directory paths without a trailing slash, which `webdav4` (used by `dvc-webdav`) does not
follow. Patch the installed package once per virtual environment:

```bash
uv run python src/patch_webdav4_redirects.py
```

##### Pushing with rclone instead of `dvc push`
For large pushes, `rclone` is a faster and resumable alternative. The DVC 3 remote layout mirrors the local cache
(`files/md5/xx/...`), so cache objects can be copied directly and are afterwards visible to `dvc pull` / `dvc status -c`.

Create the rclone remote once (reusing the password stored for DVC):

```bash
rclone config create hetzner webdav url=https://u728541.your-storagebox.de vendor=other user=u728541 \
  pass="$(uv run dvc config --local remote.hetzner.password)"
```

List the cache objects used by the workspace and copy them (run in tmux, re-run to resume):

```bash
uv run python src/dvc_used_objects.py > objects.txt
rclone copy "$(uv run dvc cache dir)/files" hetzner:mllam-exps-ShCu/files \
  --files-from objects.txt --size-only --immutable --transfers 8 --checkers 16 --progress
```

`src/dvc_used_objects.py` skips outputs whose cache cannot be collected (e.g. a missing `.dir` object) with a
warning instead of aborting like `dvc push`; fix those with `dvc commit <output>`.

## Finding sub-domains
The training/evaluation sub-domains in `data/domains.py` were originally picked by hand. The DVC stage
`find_domains` generates them programmatically for any size (`src/find_domains.py`, logic in `src/icon_domains.py`).

**Background.** ICON grids are refined by repeated bisection and store cells in that hierarchical order, so
`4**L` consecutive cells *usually* form one triangle with side `2**L` ("level `L`"; `data/domains.py` uses
`16**4 = 4**8`, i.e. level 8). In this limited-area grid, however, the lateral boundary rows (`refin_c_ctrl` 1–5)
are moved to the front of the cell list. That leaves gaps in the ordering, so fixed index offsets drift wherever
a triangle touches such a region.

**Approach.**
1. *Rebuild the triangle hierarchy bottom-up* (about 5 s). Four consecutive, index-contiguous level-`k` triangles
   are merged into a level-`k+1` triangle only if their corners really form one: of the 12 corner vertices,
   3 occur once (parent corners) and 3 occur three times (edge midpoints). Level 1 is additionally checked
   against ICON's `parent_cell_index`. Incomplete triangles at the boundary simply stay unmerged.
2. *Build domains from vertex/edge adjacency* of the complete level-`L` triangles. A domain is a 2×2 parallelogram
   of 8 triangles (a hexagon around a centre vertex plus two triangles at opposite ends), the same shape as the
   hand-made domains. There are three possible orientations, 60° apart. Only those matching the reference
   domain's orientation are kept (long diagonal ≈ 34° counter-clockwise from east for `domain01`).
3. *Select the maximum number of non-overlapping domains* exactly, as an integer program (`scipy.optimize.milp`).

The two stages:
- `get_grid_topology` downloads the needed grid variables (`vertex_of_cell`, `parent_cell_index`, …;
  about 660 MB) from DKRZ to `data/grid_topo.nc`.
- `find_domains@<level>` writes `data/domains.level<level>.py` (same format as `data/domains.py`) and the map
  `figures/domains.level<level>.png`.

The levels and the orientation reference are set in `params.yaml`:

```yaml
domain-finder:
  levels: [7, 8]
  reference-domain: domain01   # entry of data/domains.py whose orientation is used
  reference-level: 8
```

```bash
uv run dvc repro --single-item get_grid_topology find_domains@7
```

| level | cells per domain | candidates (orientation of `domain01`) | non-overlapping |
|---|---|---|---|
| 7 | 131,072 | 261 | 71 |
| 8 | 524,288 | 46 | 15 (vs. 13 hand-picked in `data/domains.py`) |

The generated files are not wired into the pipeline. `src/create_datastores.py` still imports `data/domains.py`.
To use generated domains, copy the wanted entries over (or point the import at `domains.level<level>.py`). Domains
touching the grid edge may leave too little room for the boundary ring (`margin_width_degrees`) of the boundary
datastore; check before use.

For interactive use:

```python
import sys; sys.path.insert(0, "src")
from icon_domains import TriangleTree, load_grid, select_non_overlapping
tree = TriangleTree(load_grid(cache="data/grid_topo.nc"))
ref = tree.domain_from_slices(domains["domain01"]["cells"], level=8)
doms = select_non_overlapping(tree.parallelograms(7, orientation=ref))  # also: seed=[...] to keep existing ones
tree.parallelogram_at(13.2, -57.7, level=7, orientation=ref)          # domain centred nearest a point
```

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

>[!Info]
>If a stage is failing and has been run via `--temp`, it is [currently not saved and lost](https://github.com/iterative/dvc/issues/10616).
>Run your experiment therefore in the queue or workspace.


### Restarting from a checkpoint
To continue training on a previous experiment a few steps are necessary with the current setup:
1. Checkout the experiment that shall be continued with `dvc exp apply <experiment_name/hash>`
2. Persist the checkpoints in `saved_models` by adding the `persist: true` flag to the `train` and `evaluate` stage if not yet added.
    This ensures that the checkpoints are not cleared between experiments as this would normally be done.
3. Add `--load ./saved_models/*/last.ckpt --restore_opt` to the `train.cmd` to let neural-lam start from the checkpoint
4. Patch neural_lam to read run_name from previous run, e.g. with https://github.com/mllam/neural-lam/commit/4927a5f73172a41a4f009ab3be6d36ffbec8f6dd
    This ensures that the checkpoints are updated (dvc keeps tracks of the previous one if still needed) and we are never have several
    subfolders in `saved_models` making `saved_models/*/last.ckpt` always unique.
5. Increase the number of epochs in `params.yaml`. Remember that the number of epochs are absolute and not just relative to the last checkpoint.

>[!Info]
>This stage can currently only be run in the workspace and not via `--temp` or in the `queue` [Discord-Conversation](https://discord.com/channels/485586884165107732/563406153334128681/1343195392908726272)


# Adjustments necessary for new machine

# Adjust DVC cache
Edit `.dvc/config` to point to a new DVC cache location. This can point to a local folder.