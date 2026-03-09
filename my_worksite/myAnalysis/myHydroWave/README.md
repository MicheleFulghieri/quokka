# HydroWave Convergence Study

This benchmark verifies the second-order accuracy of the Quokka code for linear hydrodynamics waves.

## Study Overview
- **Problem:** Small-amplitude linear sound wave (1D).
- **Goal:** Verify $L_1$ error convergence rate of $\mathcal{O}(N_x^{-2})$.
- **Resolutions:** $N_x \in \{32, 64, 128, 256\}$.
- **Physical Period:** $t_{max} = 1.0$ (one full sound crossing time).

## Directory Structure
- `inputs/`: Quokka input files.
- `jobs/`: Slurm batch scripts for running simulations.
- `analysis/`: Python scripts for data processing and convergence plotting.
- `results/`: Evolution plots, convergence plots, and numerical summaries.
- `logs/`: Output and error logs from Slurm jobs.

## How to Run
1. **Run Simulations:**
   ```bash
   sbatch jobs/run_convergence.slurm
   ```
2. **Analyze Results:**
   ```bash
   conda activate quokka
   python3 analysis/analyze_results.py
   ```

## Results Summary
- **Measured Convergence Order:** 2.04
- **L1 Error (Nx=256):** 5.06e-10

For a detailed explanation of the physics and syntax, see the associated documentation in the brain directory.
