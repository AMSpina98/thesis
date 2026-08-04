#!/bin/bash
#SBATCH --job-name=backprop_diversity
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s334214@studenti.polito.it
#SBATCH --partition=gpu_h200
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=backprop_diversity.log
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1

# ************************************************
# command to launch the job in the scheduler:
# sbatch launcher.sbatch

# Command to check the job status:
# squeue -u <tuo_username>
# ************************************************

module load gcc/12.4.0
module load nvhpc-hpcx-2.20-cuda12/25.1

echo "Loaded NVIDIA SDK !!!"
nvidia-smi
nvcc --version

PROJECT_DIR="$(pwd)"
BTREE_DIR="$PROJECT_DIR"

echo "Building backprop..."
cd "$BTREE_DIR" || { echo "ERROR: cannot cd into $BTREE_DIR"; exit 1; }

make clean
make

if [ ! -f "./backprop" ]; then
    echo "ERROR: build failed, executable not found"
    exit 1
fi

echo "Running backprop..."

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"

nvidia-smi -L
nvidia-smi

./backprop 65536

echo "Done!"