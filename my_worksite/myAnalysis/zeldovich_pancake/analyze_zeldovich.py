import glob
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from natsort import natsorted
from unyt import cm, km, s, Mpc
import yt

yt.set_log_level(40)  # suppress yt output except critical errors

# --- Paths ---
save_path = '/data/mfulghieri/quokka/my_worksite/myAnalysis/zeldovich_pancake/outputs'
output_dir = "/data/mfulghieri/quokka/outputs/Zeldovich/ZeldovichPancake_512_x_resolution"
plotfiles = natsorted(glob.glob(os.path.join(output_dir, "plt*")))

if not plotfiles:
    print("No plotfiles found!")
    exit(1)

def get_a(t, H0):
    # For EdS: t(a) = (2/3) * (1/H0) * a^(3/2)
    # But wait, t=0 in the simulation is a_init.
    # So t_sim = t(a) - t(a_init)
    t_init = (2.0/3.0) * (1.0/H0) * (a_init**1.5)
    t_abs = t + t_init
    a = (1.5 * H0 * t_abs)**(2.0/3.0)
    return a.value


# --- Storage for history ---
a_values             = []
times                = []
delta_values         = []   
delta_max_values     = []
gas_x, gas_y, gas_z  = [], [], []


# Storage for animations
frames_hydro  = []   # list of PNG yt files

# ----- Metadata retriving and History of the collapse -----
print("\n" + "-"*58)
print("Start analyzing the history of the gas collapse...\n")


plt.figure(figsize=(10, 6))
fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap('coolwarm_r')   

# Load all the plotfiles
for i, plt_path in enumerate(plotfiles):

    # Load the whole dataset
    ds = yt.load(plt_path)

    # Metadata retriving, scale factor and time
    metadata_file = os.path.join(plt_path, "metadata.yaml")   # load the metadata from the metadata.yaml of the plotfile

    # Convert the yaml content in a python dictionary
    if os.path.exists(metadata_file):     # if the file exists, open it in reading mode
        with open (metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)  # convert the yaml content in a python dictionary

        # Retrive the cosmological parameters
        cosmo   = metadata.get('cosmology', {})   # look for the cosmology section; if not found, return an empty dict {}
        a_init  = cosmo.get('a_init')
        a_now   = cosmo.get('a')
        z_now   = cosmo.get('z')
        H0_cgs  = cosmo.get('H0')                  # Hubble constant in cgs
        h       = cosmo.get('hubble_constant')
        omega_m = cosmo.get('Omega_m')
        omega_r = cosmo.get('Omega_r')
        omega_l = cosmo.get('Omega_Lambda')
        omega_k = cosmo.get('Omega_k')

        # Fill the dictionary of the scale factors
        a_values.append(a_now)

        # ds is the entire simulation dataset, containing grids, fields, and units.
        times.append(float(ds.current_time))  # imported directly from the simulation

    else:
        print(f"Warning: metadata not found in {plt_path}")

    
    # Load the container of all the particles
    ad = ds.all_data()

    Lx = ad.ds.domain_width[0].to('Mpc').v 
    Ly = ad.ds.domain_width[1].to('Mpc').v 
    Lz = ad.ds.domain_width[2].to('Mpc').v

    # ---- History of collapse ----

    # Creation of a line crossing the domain along x at the center (unyt_array with their unity of measure)
    c_vals = ds.domain_center.v            # domain_center returns [x_mid, y_mid, z_mid]
    left_vals = ds.domain_left_edge.v      # point where the line starts
    right_vals = ds.domain_right_edge.v    # point where the line ends
    
    start_point = [left_vals[0], c_vals[1], c_vals[2]]   # point where the line starts
    end_point   = [right_vals[0], c_vals[1], c_vals[2]]  # point where the line ends

    # Extraction of the data along the line
    ray = ds.ray(start_point, end_point)     # ray intersects the line with all 3D cells and particles in the simulation, extracting data only on that strip
   
    # Data sorting along x
    sort_idx = np.argsort(ray["index", "x"])            # index mask to growing sort the rays in x
    x_coord = ray["index", "x"][sort_idx].to('Mpc').v   # convert the coord from cm to Mpc and takes the admimensional value
    gas_density = ad['boxlib', 'gasDensity']


    # ---- 2D gas density slice ----
    slc = yt.SlicePlot(ds, "z", ("boxlib", "gasDensity"))
    slc.annotate_quiver(("boxlib", "x-GasMomentum"),
                            ("boxlib", "y-GasMomentum"), factor=14)
    slc.annotate_timestamp(corner="upper_left", draw_inset_box=True)
    gas_img_path = os.path.join(save_path, "Hydro", f"gasDensity_{i:04d}.png")
    slc.save(gas_img_path)
        
    frames_hydro.append(gas_img_path)



 # ---- Finalize history plot ----
ax_hist.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5, label=r"$\delta=0$")
ax_hist.set_title("Density contrast evolution — Zel'dovich pancake",
                      fontweight="bold", pad=12)
ax_hist.set_xlabel("x [Mpc]"); ax_hist.set_ylabel(r"$\delta$")
ax_hist.set_xlim(0, Lx)
ax_hist.grid(True, ls=":", alpha=0.5)
# Only show a subset of legend entries to avoid clutter
handles, labels = ax_hist.get_legend_handles_labels()
step = max(1, len(handles) // 8)
ax_hist.legend(handles[::step], labels[::step],
                   fontsize=8, loc="upper right", framealpha=0.85, ncol=2)
fig_hist.tight_layout()
fig_hist.savefig(os.path.join(save_path, "DeltaHistory",
                                  "delta_history.png"), dpi=300)
plt.close(fig_hist)
print(f"\nDelta history saved.")


print(f"Analysis complete. Saved plots to {save_path}")

