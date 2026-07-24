#!/usr/bin/env python3

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from datetime import datetime

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Analyze CosmoSphere Quokka run")
    p.add_argument("--save", default="./outputs",
                   help="Directory to save figures and animations")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    save_path = args.save

    # ---- Sphere radius convergence study ----
    domain_width  = 0.5   # Mpc
    rc_resolution = 256   # cells
    cell_tol      = 2     # cells
    phys_tol      = 3.91  # kpc

    print("\n" + "="*58)
    print("                    SPHERE RADIUS CONVERGENCE STUDY")
    print("="*58)
    print(f" Domain physical dimension    : {domain_width} x {domain_width} x {domain_width} Mpc")
    print(f" Domain resolution            : {rc_resolution} x {rc_resolution} x {rc_resolution} cells")
    print(f" Test spatial tolerance       : {phys_tol} kpc")
    print(f" Test cell tolerance          : {cell_tol} cells")
    print("-"*58)
    
   
    # Shifts: DM - gas
    radii       = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]) # kpc
    phys_shifts = np.array([-0.0203936, -0.01838641902, -0.01478286888, -0.006667169415,
                             0.01607885159, 0.08771846064, 0.3955204058, 1.026569249, 
                             1.67934188, 3.726723739, 6.223776142])   # kpc
    cell_shifts = np.array([-0.0104404, -0.009412863018, -0.007568038108, -0.003413234103, 
                            0.008231511932, 0.04490715965, 0.2024852907, 0.5255485427,
                            0.8597332116, 1.907883206, 3.186240465])

    # Spatial shift
    fig_rcss, ax_rcss = plt.subplots()
    ax_rcss.set_title(f'Sphere Radius Convergence', fontsize=12, fontweight='bold', y=1.04)
    ax_rcss.plot(radii, phys_shifts, '-o', markersize=3, label=f'Spatial misalignment (kpc)')
    ax_rcss.axhline(phys_tol, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Spatial tolerance (kpc)')
    ax_rcss.set_xlabel("Sphere Radius [kpc]", fontsize=11)
    ax_rcss.set_ylabel('Spatial Shift [kpc]', fontsize=11)
    ax_rcss.legend()
    ax_rcss.tick_params(direction='in', which='both')  
    ax_rcss.grid(True, linestyle='--', alpha=0.5)
    ax_rcss.set_xlim(1.01*radii[0], 0.8 * radii[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Physical_spatial_shift.png"), dpi=300)
    plt.close()

    # Cell shift
    fig_rccs, ax_rccs = plt.subplots()
    ax_rccs.set_title(f'Sphere Radius Convergence', fontsize=12, fontweight='bold', y=1.04)
    ax_rccs.plot(radii, cell_shifts, '-o', markersize=3, label=f'Spatial misalignment (cells)')
    ax_rccs.axhline(cell_tol, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Spatial tolerance (cells)')
    ax_rccs.set_xlabel("Sphere Radius [kpc]", fontsize=11)
    ax_rccs.set_ylabel('Spatial Shift [cells]', fontsize=11)
    ax_rccs.legend()
    ax_rccs.tick_params(direction='in', which='both')  
    ax_rccs.grid(True, linestyle='--', alpha=0.5)
    ax_rccs.set_xlim(1.01*radii[0], 0.8 * radii[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Cell_spatial_shift.png"), dpi=300)
    plt.close()


    # Fit of the distance and cell shift trend
    log_radii = np.log10(radii)
    log_phys_shift = np.log10(np.abs(phys_shifts))
    log_cell_shift = np.log10(np.abs(cell_shifts))

    # Linear (thanks to log10) least squares polynomial fit 
    slope_ps, intercept_ps = np.polyfit(log_radii, log_phys_shift, 1)   # 1: linear, log(shift) = intercept + slope * log(R)
    slope_cs, intercept_cs = np.polyfit(log_radii, log_cell_shift, 1) 

    # RMSE
    log_phys_pred = slope_ps * log_radii + intercept_ps  
    log_cell_pred = slope_cs * log_radii + intercept_cs   
    rmse_ps = np.sqrt(np.mean((log_phys_shift - log_phys_pred) ** 2))  
    rmse_cs = np.sqrt(np.mean((log_cell_shift - log_cell_pred) ** 2))   

    # R^2:fraction of data variance explained by the power-law model
    ss_res = np.sum((log_phys_shift - log_phys_pred)**2)            # squared sum of res: error of the model       
    ss_tot = np.sum((log_phys_shift - np.mean(log_phys_shift))**2)  # tot square sum: data variability wrt the mean
    r_squared_ps = 1 - (ss_res / ss_tot)                            # 1 - fraction of error not explained by the model
    sc_res = np.sum((log_cell_shift - log_cell_pred)**2)
    sc_tot = np.sum((log_cell_shift - np.mean(log_cell_shift))**2)
    r_squared_cs = 1 - (sc_res / sc_tot)
 
    print(f"\n          --- Radius convergence fit analysis ---")
    print(f"    Exponent of spatial shift trend:    {slope_ps}")
    print(f"    Exponent of cell shift trend   :    {slope_cs}")
    print(f"    R² log space coefficient       :    {r_squared_ps}")
    print(f"    R² log scell coefficient       :    {r_squared_cs}")
    print(f"    RMSE log space shift vs fit    :    {rmse_ps}")
    print(f"    RMSE log cell shift vs fit     :    {rmse_cs}\n")
    

    # Plot
    radii_fit       = np.linspace(radii.min(), radii.max(), 200)
    phys_shifts_fit = (radii_fit**slope_ps) * (10**(intercept_ps))

    fig_fps, ax_fps = plt.subplots()   
    ax_fps.set_title(f'Shift and Radius Converge vs LS Fit', fontsize=12, fontweight='bold', y=1.04)
    ax_fps.plot(radii, np.abs(phys_shifts), '-o', markersize=3, label=f'Spatial misalignment (module) (kpc)')
    ax_fps.plot(radii_fit, phys_shifts_fit, '-', markersize=1.5, label=f'Fit')
    ax_fps.set_xlabel("Sphere Radius [kpc]", fontsize=11)
    ax_fps.set_ylabel('|Spatial Shift| [kpc]', fontsize=11)
    ax_fps.legend()
    ax_fps.tick_params(direction='in', which='both')  
    ax_fps.grid(True, linestyle='--', alpha=0.5)
    ax_fps.set_xlim(1.01*radii[0], 0.8 * radii[-1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "Fit_spatial_shift.png"), dpi=300)
    plt.close()
    
    


                

    print(f"\nAll done. End at {datetime.now().time()}")

if __name__ == "__main__":
    main()