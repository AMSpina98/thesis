#!/bin/bash
#SBATCH --job-name=srad_diversity
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s334214@studenti.polito.it
#SBATCH --partition=gpu_h200
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=srad_diversity.log
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
SRAD_DIR="$PROJECT_DIR"

echo "Building srad..."
cd "$SRAD_DIR" || { echo "ERROR: cannot cd into $SRAD_DIR"; exit 1; }

make clean
make

if [ ! -f "./srad_v1/srad" ]; then
    echo "ERROR: build 1 failed, executable not found"
    exit 1
fi

if [ ! -f "./srad_v2/srad" ]; then
    echo "ERROR: build 2 failed, executable not found"
    exit 1
fi

echo "Running srad..."

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"

nvidia-smi -L
nvidia-smi

./srad_v1/srad 100 0.5 502 458

./srad_v2/srad 2048 2048 0 127 0 127 0.5 2

echo "Done!"