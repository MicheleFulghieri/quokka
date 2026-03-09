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
output_dir = "/data/mfulghieri/quokka/outputs/ZeldovichPancake_256_x_resolution"
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
plt.savefig("/data/mfulghieri/quokka/outputs/ZeldovichPancake_256_x_resolution/density_profiles256.png")
plt.close()

print("Analysis complete. Saved plot to /data/mfulghieri/quokka/outputs/ZeldovichPancake_256_x_resolution/density_profiles256.png")


