#!/bin/bash -l
#SBATCH --job-name=mllam-data-prep
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=cu_0003
#SBATCH --output=/leonardo/home/userexternal/%u/logs/neurallam.%j.log
#SBATCH --error=/leonardo/home/userexternal/%u/logs/neurallam.%j.log

cd ${DVC_WORKING_DIR}
echo "Started slurm job $SLURM_JOB_ID"

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

# Export for stability
export OMPI_MCA_coll_hcoll_enable=0

echo "Using venv in ${MLLAM_VENV_PATH}"

# source the virtual environment so that the python script can be run
source ${MLLAM_VENV_PATH}/bin/activate

# the config path is the first argument for mllam_data_prep
CONFIG_PATH=$1
# assume third argument is the output path
OUTPUT_PATH=$3

# Use a python script (which internally uses mlflow) to run a background process that logs system metrics.
# Explanation:
# - `python log_system_metrics.py &`  
#     -> Starts the Python script as a background process.
#     -> The `&` ensures it runs in the background and doesn't block the terminal.
# - `LOGGER_PID=$!`  
#     -> Captures the Process ID (PID) of the last background process (`$!` refers to the last process run in the background).
#     -> This allows us to reference the process later (e.g., to stop it with `kill $LOGGER_PID`).
# - `disown`  
#     -> Removes the background process from the shell’s job table.
#     -> Prevents the process from being terminated if the shell session is closed.
#     -> Output from the script remains visible in the terminal.

python machines/log_system_metrics_during_dataprep.py --config_path=${CONFIG_PATH} --dataset_output_path=${OUTPUT_PATH} & LOGGER_PID=$! && disown

srun -ul -K1 python -m mllam_data_prep "$@"

# kill the logger process, ensuring it has stopped before exiting
while ps -p $LOGGER_PID > /dev/null; kill $LOGGER_PID; do sleep 1; done
