import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from unyt import Mpc, km, s, g, cm

# Parameters (consistent with ZeldovichPancake.in)
h = 0.7
H0 = (h * 100.0 * km / s / Mpc).to('1/s')
L = 20.0 * Mpc
a_init = 0.02

def get_a(t):
    # For EdS: t(a) = (2/3) * (1/H0) * a^(3/2)
    # But wait, t=0 in the simulation is a_init.
    # So t_sim = t(a) - t(a_init)
    t_init = (2.0/3.0) * (1.0/H0) * (a_init**1.5)
    t_abs = t + t_init
    a = (1.5 * H0 * t_abs)**(2.0/3.0)
    return a.value

# Path to plotfiles
output_dir = "/data/mfulghieri/quokka/outputs/ZeldovichPancake_11237"
plotfiles = sorted(glob.glob(os.path.join(output_dir, "plt*")))

if not plotfiles:
    print("No plotfiles found!")
    exit(1)

plt.figure(figsize=(10, 6))

for plt_file in plotfiles:
    ds = yt.load(plt_file)
    t = ds.current_time.to('s')
    a = get_a(t)
    
    # Get 1D profile along x-axis
    # We take a line through the center of the box
    left = ds.domain_left_edge
    right = ds.domain_right_edge
    center = (left + right) / 2.0
    
    # Create a ray along x
    ray = ds.ray(left, [right[0], center[1], center[2]])
    
    sort_idx = np.argsort(ray['x'])
    x = ray['x'][sort_idx].to('Mpc')
    rho = ray['boxlib', 'gasDensity'][sort_idx]
    
    rho_mean = np.mean(rho)
    delta = (rho / rho_mean) - 1.0
    
    plt.plot(x, rho, label=f"a = {a:.3f}")
    
    print(f"File: {os.path.basename(plt_file)}, t = {t:.2e}, a = {a:.3f}, rho_max/rho_min = {np.max(rho)/np.min(rho):.2f}")

plt.xlabel("x [Mpc]")
plt.ylabel("Density [g/cm^3]")
plt.title("Zel'dovich Pancake: Density Profile Evolution")
plt.legend()
plt.grid(True)
plt.savefig("/data/mfulghieri/quokka/analysis/zeldovich_pancake/density_profiles.png")
plt.close()

print("Analysis complete. Saved plot to /data/mfulghieri/quokka/analysis/zeldovich_pancake/density_profiles.png")
