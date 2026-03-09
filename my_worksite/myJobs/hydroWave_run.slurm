#!/bin/bash 
set -euo pipefail        # exit on error, undefined variable, and fail on pipe errors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8000mb
#SBATCH --time=12:00:00
#SBATCH -J HydroWave
#SBATCH -p long
#SBATCH -o /data/mfulghieri/quokka/outputs/HydroWave_logs/%j.out
#SBATCH -e /data/mfulghieri/quokka/outputs/HydroWave_logs/%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=m.fulghieri@campus.unimib.it

# Useful paths
BASE_DIR="/data/mfulghieri/quokka"
EXE="${BASE_DIR}/build/src/problems/HydroWave/HydroWave"
INPUT="${BASE_DIR}/inputs/HydroWave.in"
# We'll use a unique output directory per job
OUT_DIR="${BASE_DIR}/outputs/HydroWave_${SLURM_JOB_ID}"

# Creation of the required directories
mkdir -p "${OUT_DIR}"
mkdir -p /data/mfulghieri/quokka/outputs/HydroWave_logs

# Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quokka

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK

# Execution
echo "Starting HydroWave simulation at: $(date)"

# Print some useful information about the job
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
sacct -j $SLURM_JOB_ID --format=JobID,State,Elapsed,MaxRSS

# Command execution
srun $EXE $INPUT --plotfile_prefix="$OUT_DIR/plt"  # srun is used to launch the executable in a way that is compatible with SLURM's resource management

echo "Simultion finished at: $(date)"

