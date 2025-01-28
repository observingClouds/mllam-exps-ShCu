module load GCC/12.3.0
module load OpenMPI/4.1.5
module load SciPy-bundle/2024.05
module load foss/2023a
# module load Neural-LAM/0.2.0-PyTorch-2.3.1-CUDA-12.4.0
module load GCCcore/12.3.0
module load NCCL/2.18.3-CUDA-12.4.0
# module load wandb/0.16.6
module load slurm

# wandb off

# export PYTHONPATH=/dcai/projects/cu_0003/user_space/hinkas/neural-lam:$PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/dcai/projects/cu_0003/user_space/skc/packages/

source /dcai/users/schhau/git-repos/neural-lam/.venv/bin/activate
