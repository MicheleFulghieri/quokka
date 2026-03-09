import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from unyt import cm, km, s, g, erg

# --- Configuration ---
analysis_dir = "/data/mfulghieri/quokka/benchmarks/HydroWave/results"
os.makedirs(analysis_dir, exist_ok=True)

resolutions = [32, 64, 128, 256]
l1_errors = []

print("Starting refined analysis and convergence study...")

# 1. Profile Evolution Plot (for a single resolution, e.g., 256)
ref_res = 256
output_dir_ref = f"/data/mfulghieri/quokka/outputs/HydroWave_N{ref_res}"
plotfiles_ref = sorted(glob.glob(os.path.join(output_dir_ref, "plt*")))

if plotfiles_ref:
    print(f"Generating evolution plot for Nx={ref_res}...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    
    # Decimate: plot 6 snapshots total
    n_plots = 6
    idx_list = np.linspace(0, len(plotfiles_ref)-1, n_plots, dtype=int)
    colors = plt.cm.plasma(np.linspace(0, 1, n_plots))
    
    for i, idx in enumerate(idx_list):
        plt_file = plotfiles_ref[idx]
        ds = yt.load(plt_file)
        t_sim = ds.current_time.to('s').value
        
        ad = ds.all_data()
        ray = ds.ray(ds.domain_left_edge, [ds.domain_right_edge[0], ds.domain_center[1], ds.domain_center[2]])
        sort_idx = np.argsort(ray['x'])
        x = ray['x'][sort_idx].to('cm').value
        
        rho = ray[('boxlib', 'gasDensity')][sort_idx].to('g/cm**3').value
        xmom = ray[('boxlib', 'x-GasMomentum')][sort_idx]
        v_x = (xmom / ray[('boxlib', 'gasDensity')][sort_idx]).to('km/s').value
        e_int = ray[('boxlib', 'gasInternalEnergy')][sort_idx]
        gamma = 5.0/3.0
        pressure = (e_int * (gamma - 1)).to('erg/cm**3').value
        
        label = f"t = {t_sim:.2f} s"
        axes[0].plot(x, rho, label=label, color=colors[i], lw=2)
        axes[1].plot(x, v_x, color=colors[i], lw=2)
        axes[2].plot(x, pressure, color=colors[i], lw=2)

    axes[0].set_ylabel(r"Density $\rho$ [g/cm$^3$]")
    axes[0].legend(loc='upper right', fontsize='medium', ncol=2)
    axes[0].set_title(f"HydroWave Test: Profile Evolution (Nx={ref_res})", fontsize=16)
    axes[1].set_ylabel(r"Velocity $v_x$ [km/s]")
    axes[1].axhline(0, color='black', ls='--', alpha=0.3)
    axes[2].set_ylabel(r"Pressure $P$ [erg/cm$^3$]")
    axes[2].set_xlabel("x [cm]")
    for ax in axes:
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.set_xlim(x.min(), x.max())

    plt.savefig(os.path.join(analysis_dir, "hydrowave_refined_evolution.png"), dpi=300, bbox_inches='tight')
    print("Refined evolution plot saved.")

# 2. Convergence Study
print("Performing convergence analysis...")
valid_resolutions = []
for res in resolutions:
    out_dir = f"/data/mfulghieri/quokka/outputs/HydroWave_N{res}"
    pfiles = sorted(glob.glob(os.path.join(out_dir, "plt*")))
    
    if len(pfiles) >= 2:
        ds0 = yt.load(pfiles[0])
        dsN = yt.load(pfiles[-1])
        
        ad0 = ds0.all_data()
        adN = dsN.all_data()
        
        rho0 = ad0[('boxlib', 'gasDensity')].value
        rhoN = adN[('boxlib', 'gasDensity')].value
        
        l1_err = np.mean(np.abs(rhoN - rho0))
        l1_errors.append(l1_err)
        valid_resolutions.append(res)
        print(f"Nx = {res}: L1 Error = {l1_err:.2e}")
    else:
        print(f"Warning: Not enough plotfiles for Nx = {res}. Skipping.")

if len(valid_resolutions) >= 2:
    plt.figure(figsize=(8, 6))
    plt.loglog(valid_resolutions, l1_errors, 'o-', lw=2, markersize=8, label="Numerical Error")
    
    # Add reference slope (2nd order)
    res_array = np.array(valid_resolutions)
    ref_slope = l1_errors[0] * (res_array / res_array[0])**-2
    plt.loglog(res_array, ref_slope, 'k--', alpha=0.6, label=r"Reference ($N_x^{-2}$)")
    
    plt.xlabel(r"Resolution $N_x$", fontsize=12)
    plt.ylabel(r"$L_1$ Error Norm (Density)", fontsize=12)
    plt.title("HydroWave Convergence Study", fontsize=14)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.savefig(os.path.join(analysis_dir, "hydrowave_convergence.png"), dpi=300, bbox_inches='tight')
    print("Convergence plot saved.")
    
    # Calculate measured order of convergence
    orders = []
    for i in range(len(l1_errors)-1):
        order = -np.log(l1_errors[i+1]/l1_errors[i]) / np.log(valid_resolutions[i+1]/valid_resolutions[i])
        orders.append(order)
    avg_order = np.mean(orders)
    print(f"Measured average order of convergence: {avg_order:.2f}")

    with open(os.path.join(analysis_dir, "convergence_summary.txt"), "w") as f:
        f.write("HydroWave Convergence Summary\n")
        f.write("-----------------------------\n")
        f.write("Resolution Nx | L1 Error\n")
        for r, e in zip(valid_resolutions, l1_errors):
            f.write(f"{r:13d} | {e:.2e}\n")
        f.write(f"\nMeasured Average Order: {avg_order:.2f}\n")

print("Analysis pipeline finished.")
