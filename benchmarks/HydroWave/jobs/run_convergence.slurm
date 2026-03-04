#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8000mb
#SBATCH --time=12:00:00
#SBATCH -J HydroWaveConvergence
#SBATCH -p long
#SBATCH -o /data/mfulghieri/quokka/benchmarks/HydroWave/logs/conv_%j.out
#SBATCH -e /data/mfulghieri/quokka/benchmarks/HydroWave/logs/conv_%j.err

# Useful paths
BASE_DIR="/data/mfulghieri/quokka"
EXE="${BASE_DIR}/build/src/problems/HydroWave/HydroWave"
INPUT="${BASE_DIR}/benchmarks/HydroWave/inputs/HydroWave.in"

# Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quokka

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK

# Resolutions to test
RESOLUTIONS=(32 64 128 256)

for NX in "${RESOLUTIONS[@]}"
do
    OUT_DIR="${BASE_DIR}/outputs/HydroWave_N${NX}"
    mkdir -p "${OUT_DIR}"
    echo "Starting HydroWave simulation with Nx=${NX} at: $(date)"
    
    # Run simulation overriding resolution
    # amr.n_cell = NX 8 8
    # amr.max_grid_size = NX
    $EXE $INPUT amr.n_cell="${NX} 8 8" amr.max_grid_size="${NX}" plotfile_prefix="${OUT_DIR}/plt"
    
    echo "Finished resolution Nx=${NX} at: $(date)"
done

echo "Convergence study finished at: $(date)"
