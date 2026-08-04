#!/bin/bash
#SBATCH --job-name=btree_diversity
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s334214@studenti.polito.it
#SBATCH --partition=gpu_v100
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=btree_diversity.log
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

# Percorso della cartella del progetto (assumo che lo sbatch sia lanciato
# dalla root di "b+tree", altrimenti sostituisci con il percorso assoluto)
PROJECT_DIR="$(pwd)"
BTREE_DIR="$PROJECT_DIR"

echo "Building b+tree..."
cd "$BTREE_DIR" || { echo "ERROR: cannot cd into $BTREE_DIR"; exit 1; }

make clean
make

if [ ! -f "./b+tree.out" ]; then
    echo "ERROR: build failed, executable not found"
    exit 1
fi

echo "Running b+tree..."

# Adatta questi due path ai file reali nella tua cartella "data"
INPUT_FILE="$PROJECT_DIR/data/mil.txt"
COMMAND_FILE="$PROJECT_DIR/data/command.txt"

./b+tree.out file "$INPUT_FILE" command "$COMMAND_FILE"

echo "Done!"