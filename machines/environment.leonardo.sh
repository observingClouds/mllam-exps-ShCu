module load gcc
module load openmpi
module load nccl
module load cuda

#export CARTOPY_DATA_DIR=/dcai/projects01/cu_0003/user_space/has/cartopy_features
export MLFLOW_TRACKING_URI="sqlite:////leonardo/home/userexternal/hschulz0/mlflow/mlflow.db"
export MLFLOW_TRACKING_INSECURE_TLS=true

# wandb off
