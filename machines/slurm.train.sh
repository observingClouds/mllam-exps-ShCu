#!/bin/bash -l
#SBATCH --job-name=HAS-NeuralLam
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=logs/neurallam.%j.log
#SBATCH --error=logs/neurallam.%j.log

echo "Started slurm job $SLURM_JOB_ID"

# uv venv nlm
# source nlm/bin/activate
# uv pip install neural-lam dvc dvclive
source ~/git-repos/nlm/bin/activate


wandb disabled

set -a
LOGLEVEL=INFO
CUDA_LAUNCH_BLOCKING=1

OMPI_MCA_pml=ucx
OMPI_MCA_btl=^vader,tcp,openib,uct
UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
NCCL_SOCKET_IFNAME=ens6f0
NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
OMP_NUM_THREADS=56
set +a

srun -ul python train_wrapper.py "$@"
