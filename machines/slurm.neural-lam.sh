#!/bin/bash -l
#SBATCH --output=/leonardo/home/userexternal/%u/logs/neurallam.%j.log
#SBATCH --error=/leonardo/home/userexternal/%u/logs/neurallam.%j.log

# chdir in the slurm directive is to make sure the gpu stats file is saved there
# Then we cd to the actual working directory

cd ${DVC_WORKING_DIR}

#echo "Started slurm job $SLURM_JOB_ID"

# Get the hostname
HOSTNAME=$(hostname)

# Flag to check if any script is sourced
SOURCED=false

# Loop through all environment scripts
for SCRIPT in machines/environment.*.sh; do
    # Extract the base name from the script name (e.g., 'leonardo' from 'environment.leonardo.sh')
    BASE_NAME=$(basename "$SCRIPT" | cut -d '.' -f 2)

    # Check if the base name is part of the hostname
    if [[ "$HOSTNAME" == *"$BASE_NAME"* ]]; then
        echo "Sourcing $SCRIPT for hostname $HOSTNAME"
        source "$SCRIPT"
        SOURCED=true
    fi
done

# If no script was sourced, print a message
if ! $SOURCED; then
    echo "No matching environment script found for hostname $HOSTNAME"
fi

set -a
LOGLEVEL=INFO
#CUDA_LAUNCH_BLOCKING=1

#OMPI_MCA_pml=ucx
#OMPI_MCA_btl=^vader,tcp,openib,uct
#UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
#NCCL_SOCKET_IFNAME=ens6f0
#NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
#OMP_NUM_THREADS=56
#OMPI_MCA_coll_hcoll_enable=0
set +a

echo "Using venv in ${MLLAM_VENV_PATH}"

# source the virtual environment so that the python script can be run
source ${MLLAM_VENV_PATH}/bin/activate

# Check if 'eval' is in the arguments
if [[ " $@ " == *" --eval "* ]]; then
        MODE="eval"
    else
        MODE="train"
fi

# pass all arguments to the python script
NCCL_DEBUG=INFO
python -m neural_lam.train_model --logger_run_name $MODE-$DVC_EXP_NAME "$@"
