module load gcc
module load openmpi
module load nccl
module load cuda

#export CARTOPY_DATA_DIR=/dcai/projects01/cu_0003/user_space/has/cartopy_features
export MLFLOW_TRACKING_URI="https://mlflow.dmidev.org"
export MLFLOW_TRACKING_INSECURE_TLS=true
export MACHINE_PREFIX="srun -ul -K1 "

# wandb off
