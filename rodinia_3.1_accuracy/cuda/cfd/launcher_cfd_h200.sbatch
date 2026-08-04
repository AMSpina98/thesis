#!/bin/bash
#SBATCH --job-name=cfd_diversity
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s334214@studenti.polito.it
#SBATCH --partition=gpu_h200
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=cfd_diversity.log
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1

# ************************************************
# command to launch the job in the scheduler:
# sbatch launcher.sbatch

# Command to check the job status:
# squeue -u <tuo_username>
# ************************************************

module load gcc/12.4.0
module load hpcx
module load nvhpc-hpcx-2.20-cuda12/25.1

echo "Loaded NVIDIA SDK !!!"
nvidia-smi
nvcc --version

PROJECT_DIR="$(pwd)"
CFD_DIR="$PROJECT_DIR"

echo "Building cfd..."
cd "$CFD_DIR" || { echo "ERROR: cannot cd into $CFD_DIR"; exit 1; }

make clean
make

if [ ! -f "./euler3d" ]; then
    echo "ERROR: build failed, executable not found"
    exit 1
fi

echo "Running cfd..."

# Adatta questi due path ai file reali nella tua cartella "data"
INPUT_FILE="$PROJECT_DIR/../../data/cfd/fvcorr.domn.097K"

./euler3d "$INPUT_FILE"

echo "Done!"