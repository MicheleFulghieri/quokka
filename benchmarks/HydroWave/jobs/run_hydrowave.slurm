#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8000mb
#SBATCH --time=12:00:00
#SBATCH -J HydroWave
#SBATCH -p long
#SBATCH -o /data/mfulghieri/quokka/benchmarks/HydroWave/logs/%j.out
#SBATCH -e /data/mfulghieri/quokka/benchmarks/HydroWave/logs/%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=m.fulghieri@campus.unimib.it

# Useful paths
BASE_DIR="/data/mfulghieri/quokka"
EXE="${BASE_DIR}/build/src/problems/HydroWave/HydroWave"
INPUT="${BASE_DIR}/benchmarks/HydroWave/inputs/HydroWave.in"

# Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quokka

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK

# We'll use a unique output directory per job
OUT_DIR="${BASE_DIR}/outputs/HydroWave_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

# Execution
echo "Starting HydroWave simulation at: $(date)"

# Command execution
$EXE $INPUT --plotfile_prefix="$OUT_DIR/plt"

echo "Simulation finished at: $(date)"
