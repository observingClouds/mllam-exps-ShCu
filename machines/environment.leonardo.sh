module load gcc
module load openmpi
module load nccl
module load cuda

#export CARTOPY_DATA_DIR=/dcai/projects01/cu_0003/user_space/has/cartopy_features
export MLFLOW_TRACKING_URI="sqlite:////leonardo_work/DestE_330_25/users/hschulz0/repos/mllam-exps-ShCu/mlflow.$DVC_EXP_NAME.db" #"https://mlflow.dmidev.org"
export MLFLOW_TRACKING_INSECURE_TLS=true
export MACHINE_PREFIX="srun -ul -K1 "
# https://docs-31e4a2.pages.it4i.eu/articles/best-practices/leonardo-ai-workloads/#1-install-the-smallstep-client
export NCCL_NET="IB"
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=29500
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5
export NCCL_IB_ENABLE=1

# wandb off
