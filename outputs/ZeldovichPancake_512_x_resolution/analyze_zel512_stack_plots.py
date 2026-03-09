import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from unyt import Mpc, km, s, K, g, cm

# --- Configuration ---
h = 0.7
H0 = (h * 100.0 * km / s / Mpc).to('1/s')
a_init = 0.02
output_dir = "/data/mfulghieri/quokka/outputs/ZeldovichPancake_512_x_resolution"
plotfiles = sorted(glob.glob(os.path.join(output_dir, "plt*")))

def get_a(t_sim):
    """Get the scale factor a(t) for Einstein-de Sitter."""
    t_init = (2.0/3.0) * (1.0/H0) * (a_init**1.5)
    t_abs = t_sim + t_init  # quokka start the time from 0, but the universe has already an age t(a_init)
    return (1.5 * H0 * t_abs)**(2.0/3.0)

if not plotfiles:
    print("Error: no plotfile found!")
    exit(1)

# Select some meaningful steps (e. g. start, half, end)
#indices = [0, len(plotfiles)//2, len(plotfiles)-1]   # // rounds the division to the closest integer
#selected_files = [plotfiles[i] for i in indices]

# Setup of the figures
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.05)

for plt_file in plotfiles:
    ds = yt.load(plt_file)    # load the dataset
    a = get_a(ds.current_time).value
    
    # 1D data extraction
    ad = ds.all_data()
    # creation of a ray: a sampling line(way to analyze 1D problem in a 3D grid)
    left_edge = ds.domain_left_edge
    right_edge = ds.domain_right_edge
    ray = ds.ray(left_edge, [right_edge[0], ds.domain_center[1], ds.domain_center[2]])
    
    sort_idx = np.argsort(ray['x']) # yt return the cell data not order, but according to the processor distribution -> space rearrangement of the data before plot 
    x_mpc = ray['x'][sort_idx].to('Mpc').value  # sort the x coordinates and convert to Mpc

   # Saved data (from print(ds.field_list)):
   # [('boxlib', 'gasDensity'), ('boxlib', 'gasEnergy'), ('boxlib', 'gasInternalEnergy'), ('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), ('boxlib', 'z-GasMomentum')]

    
    # 1. Density (Overdensity)
    rho = ray[('boxlib', 'gasDensity')][sort_idx]
    rho_mean = np.mean(rho)
    overdensity = (rho / rho_mean).value
    
    # 2. Peculiar velocity (v_phys / a)
    x_mom = ray[('boxlib', 'x-GasMomentum')][sort_idx]
    v_pec = (x_mom / rho).to('km/s').value
    
    # 3. Temperature
    e_int = ray[('boxlib', 'gasInternalEnergy')][sort_idx]
    gamma = 5.0/3.0
    mu = 1.0  # assuming only hydrogen for the pancake
    m_u = 1.660539e-24 * g
    k_B = 1.380649e-16 * (g * cm**2 / (s**2 * K))
    
    # Note: e_int in Quokka is energy density (erg/cm^3 comoving)
    # rho is mass density (g/cm^3 comoving)
    temp = (e_int * (gamma - 1) * mu * m_u) / (rho * k_B)
    temp = temp.to('K').value

    # --- Plotting ---
    label = f"a = {a:.3f}"
    axes[0].plot(x_mpc, overdensity, label=label, lw=2)
    axes[1].plot(x_mpc, v_pec, lw=2)
    axes[2].plot(x_mpc, temp, lw=2)

# --- plot refinement ---
axes[0].set_ylabel(r"Overdensity $\rho / \bar{\rho}$")
axes[0].set_yscale('log')
axes[0].legend(loc='upper right')
axes[0].set_title("Zel'dovich Pancake (Quokka RHD)")

axes[1].set_ylabel(r"$v_{pec}$ [km/s]")
axes[1].axhline(0, color='black', ls='--', alpha=0.5) # reference for the collapse

axes[2].set_ylabel("Temperature [K]")
axes[2].set_yscale('log')
axes[2].set_xlabel("x [Mpc]")

for ax in axes:
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(x_mpc.min(), x_mpc.max())

plt.savefig("/data/mfulghieri/quokka/outputs/ZeldovichPancake_512_x_resolution/pancake_physics_stack521.png", dpi=300)
print(f"Analysis completed. Plots saved")
