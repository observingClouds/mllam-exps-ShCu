#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=30G
#SBATCH --gres=tmpfs:200G
#SBATCH -t 3:30:00
# Compute metrics for all prediction datasets

# Array of prediction datasets (path and name)
declare -A DATASETS=(
    ["../evals/rain_unroll30.zarr"]="rr"
    # ["../evals/baseline_unroll30.zarr"]="baseline"
    # ["../evals/bare-minimum_unroll30.zarr"]="bare-minimum"
    # ["../evals/lhfl_unroll30.zarr"]="lhfl"
    # ["../evals/lw_unroll30.zarr"]="lw"
    # ["../evals/qv_unroll30.zarr"]="qv"
    # ["../evals/sw_unroll30.zarr"]="sw"
    # ["../evals/swlw_unroll30.zarr"]="swlw"
    #["../evals/sfconly_unroll30.zarr"]="sfconly"
)

source .venv/bin/activate
# Compute metrics for each dataset
for path in "${!DATASETS[@]}"; do
    name="${DATASETS[$path]}"
    echo "Computing metrics for $name..."
    python compute_metrics.py "$path" "$name"
done

echo "All metrics computed!"
echo ""
echo "To visualize results, run:"
echo "python visualize_metrics.py metrics_*.nc"
