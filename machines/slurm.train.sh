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

module load \
    GCC/12.3.0 \
    OpenMPI/4.1.5 \
    SciPy-bundle/2024.05 \
    Neural-LAM/0.2.0-PyTorch-2.3.1-CUDA-12.4.0 \
    wandb/0.16.6 \
    foss/2023a \
    GCCcore/12.3.0 \
    NCCL/2.18.3-CUDA-12.4.0

#pip install parse --target packages --no-deps
export PYTHONPATH=$HOME/git-repos/neural-lam:$PYTHONPATH
export PYTHONPATH=/dcai/projects/cu_0003/user_space/has/packages:$PYTHONPATH
export PYTHONPATH=/dcai/projects/cu_0003/user_space/has/packages_dvc:$PYTHONPATH


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
