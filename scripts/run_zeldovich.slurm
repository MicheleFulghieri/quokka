#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8000mb
#SBATCH --time=12:00:00
#SBATCH -J Quokka_Zeldovich
#SBATCH -p long
#SBATCH -o /data/mfulghieri/quokka/outputs/ZeldovichPancake_logs/%j.out
#SBATCH -e /data/mfulghieri/quokka/outputs/ZeldovichPancake_logs/%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=m.fulghieri@campus.unimib.it

# Useful paths
BASE_DIR="/data/mfulghieri/quokka"
EXE="${BASE_DIR}/build/src/problems/ZeldovichPancake/ZeldovichPancake"
INPUT="${BASE_DIR}/inputs/ZeldovichPancake.in"
# We'll use a unique output directory per job
OUT_DIR="${BASE_DIR}/outputs/ZeldovichPancake_${SLURM_JOB_ID}"

# Creation of the required directories
mkdir -p "${OUT_DIR}"
mkdir -p /data/mfulghieri/quokka/outputs/ZeldovichPancake_logs

# Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quokka

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK

# Execution
echo "Inizio simulazione Zel'dovich Pancake alle: $(date)"

# Command execution
$EXE $INPUT --amrex.plot_file="$OUT_DIR/plt"

echo "Simulazione terminata alle: $(date)"
