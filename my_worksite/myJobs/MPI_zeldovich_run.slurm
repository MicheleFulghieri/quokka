#!/bin/bash 

#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000mb
#SBATCH --time=12:00:00
#SBATCH -J Zeldovich
#SBATCH -p long
#SBATCH -o /data/mfulghieri/quokka/outputs/ZeldovichPancake_logs/job_%j.out
#SBATCH -e /data/mfulghieri/quokka/outputs/ZeldovichPancake_logs/job_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=m.fulghieri@campus.unimib.it

# Useful paths
BASE_DIR="/data/mfulghieri/quokka"
EXE="${BASE_DIR}/build/src/problems/ZeldovichPancake/ZeldovichPancake"
INPUT="${BASE_DIR}/inputs/ZeldovichPancake.in"
OUT_DIR="${BASE_DIR}/outputs/ZeldovichPancake_${SLURM_JOB_ID}"

# Creation of the required directories
mkdir -p "${OUT_DIR}"
mkdir -p /data/mfulghieri/quokka/outputs/ZeldovichPancake_logs

# Environment
source /data/mfulghieri/anaconda3/etc/profile.d/conda.sh   # Initialize conda for bash
eval "$(conda shell.bash hook)"                            # Activate the quokka environment
conda activate quokka                                      # Activate the quokka environment, which should contain all necessary dependencies

# Set OpenMP environment variables based on SLURM settings
export OMP_NUM_THREADS=1                     # Set the number of OpenMP threads to match the allocated CPUs
export OMP_PLACES=threads                    # Distribute threads across CPU cores
export OMP_PROC_BIND=spread                  # Spread threads across cores to improve performance  

# exit on error 
set -e        # Exit immediately if a command exits with a non-zero status

# Execution
echo "Start Zel'dovich Pancake simulation: $(date) on $(hostname)"
echo "Resources: Tasks=1, CPUs=$SLURM_CPUS_PER_TASK"

# Print some useful information about the job
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# Change to the output directory before running the command to ensure all output files are created there
cd "${OUT_DIR}"

# Command execution
srun -n $SLURM_NTASKS $EXE $INPUT --plotfile_prefix="$OUT_DIR/plt"

echo "Simulation terminated at: $(date)"

sacct -j $SLURM_JOB_ID --format=JobID,State,Elapsed,MaxRSS
