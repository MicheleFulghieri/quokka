import yt
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Anti-Grain Geometry, backend without interface, allow to save the plots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # for the color map
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
import os
import glob
import yaml
from natsort import natsorted
from unyt import Mpc, km, s, g, cm


# Set yt's verbosity 
yt.set_log_level(40)   # level 40 only displays critical errors


# Useful paths
plotfiles_dir = "/data/mfulghieri/quokka/outputs/SphereDMCosmo/SphereDMCosmo_12497"
save_path = '/data/mfulghieri/quokka/my_worksite/myAnalysis/SphereDMCosmo/outputs'

# # Create output folder if not already existent
# os.makedirs(os.path.join(save_path, "DensityProfiles"), exist_ok=True)
# os.makedirs(os.path.join(save_path, "GasVsDM"), exist_ok=True)

# Retrive the plotfiles 
# os.path.join joins the folder path with the "plt*" pattern, glob.glob creates a list of all files/folders that begin with "plt"
plotfiles = glob.glob(os.path.join(plotfiles_dir, "plt*"))
#plotfiles = [f for f in glob.glob(os.path.join(plotfiles_dir, "plt[0-9]*")) if ".old" not in f]
plotfiles = natsorted(plotfiles)     # natural ordering to the list 

if not plotfiles: 
    print("No plotfiles found!")
    exit(1)


# --- Scale factors and times ---

# Dictionaries to stores the evolution
a_values  = []
z_values  = []
times     = []
dm_x      = []
dm_y      = []
dm_z      = []
gas_x     = []
gas_y     = []
gas_z     = []


# ----- Metadata retriving and History of the collapse -----
print("\n" + "-"*58)
print("Start analyzing the cosmological gas e particle drift...\n")


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

        # Fill the dictionary of the scale factors and redshifts
        a_values.append(a_now)
        z_values.append(z_now)

        # ds is the entire simulation dataset, containing grids, fields, and units.
        times.append(float(ds.current_time))  # imported directly from the simulation

    else:
        print(f"Warning: metadata not found in {plt_path}")

    
    # Load the container of all the particles
    ad = ds.all_data()


    # ---- Tracking positions particle and gas ----

    # Particle
    x_dm_arr = ad['CIC_particles', "particle_position_x"].to('Mpc').v
    y_dm_arr = ad['CIC_particles', "particle_position_y"].to('Mpc').v
    z_dm_arr = ad['CIC_particles', "particle_position_z"].to('Mpc').v
    
    if len(x_dm_arr) > 0:
        dm_x.append(float(x_dm_arr[0]))   # 0 since only 1 particle
        dm_y.append(float(y_dm_arr[0]))
        dm_z.append(float(z_dm_arr[0]))
    else:
        dm_x.append(np.nan)
        dm_y.append(np.nan)
        dm_z.append(np.nan)


    gas_density = ad['boxlib', 'gasDensity']
    x_coordinates = ad['index', 'x'].to('Mpc').v
    y_coordinates = ad['index', 'y'].to('Mpc').v
    z_coordinates = ad['index', 'z'].to('Mpc').v

    # Mean position weighted on the density (track sphere center in 3D)
    total_gas_density = np.sum(gas_density)
    if total_gas_density > 0:
        gas_center_x = np.sum(gas_density * x_coordinates) / total_gas_density
        gas_center_y = np.sum(gas_density * y_coordinates) / total_gas_density
        gas_center_z = np.sum(gas_density * z_coordinates) / total_gas_density
        
        gas_x.append(float(gas_center_x))
        gas_y.append(float(gas_center_y))
        gas_z.append(float(gas_center_z))
    else:
        gas_x.append(np.nan)
        gas_y.append(np.nan)
        gas_z.append(np.nan)


    print(f"Plotfile {i:03d} | a = {a_now:.4f} | z = {z_now:.2f}")
    print(f"  -> DM  (X,Y,Z): ({dm_x[-1]:.4f}, {dm_y[-1]:.4f}, {dm_z[-1]:.4f}) Mpc")
    print(f"  -> Gas (X,Y,Z): ({gas_x[-1]:.4f}, {gas_y[-1]:.4f}, {gas_z[-1]:.4f}) Mpc")

    
    # ---- Hydro analysis ----

    # Density
    slc = yt.SlicePlot(ds, 'z', ('boxlib', 'gasDensity'))
    slc.annotate_quiver(('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), factor=12)
    slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
    slc.set_zlim(('boxlib', 'gasDensity'), 1e-31, 1e-20)  # colorbar CGS range
    slc.annotate_particles(width=(10.0, 'Mpc'), p_size=40, col='white', marker='o', ptype='CIC_particles')
    slc.annotate_title("Density")
    slc.annotate_scale(corner="upper_right")
    text_legend = r"$\circ$ White Circle: DM Particle (CIC)"
    slc.annotate_text((0.05, 0.05), text_legend, coord_system="axis")


    slc.save(os.path.join(save_path + "/Hydro/Density", f"Density_{i:03d}.png"))

    # Internal energy
    slc = yt.SlicePlot(ds, 'z', ('boxlib', 'gasInternalEnergy'))
    slc.annotate_quiver(('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), factor=12)
    slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
    slc.set_zlim(('boxlib', 'gasInternalEnergy'), 1e-29, 1e-16)  # colorbar CGS range
    slc.annotate_scale(corner="upper_right")

    slc.save(os.path.join(save_path + "/Hydro/Eint", f"InternalEnergy_{i:03d}.png"))

    # x-Momentum
    slc = yt.SlicePlot(ds, 'z', ('boxlib', 'x-GasMomentum'))
    slc.annotate_quiver(('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), factor=12)
    slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
    slc.annotate_scale(corner="upper_right")

    slc.save(os.path.join(save_path + "/Hydro/XMomentum", f"XMomentum_{i:03d}.png"))


    # y-Momentum
    slc = yt.SlicePlot(ds, 'z', ('boxlib', 'y-GasMomentum'))
    slc.annotate_quiver(('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), factor=12)
    slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
    slc.annotate_scale(corner="upper_right")

    slc.save(os.path.join(save_path + "/Hydro/YMomentum", f"YMomentum_{i:03d}.png"))

    ds.index.clear_all_data()  # free the RAM


# Numpy array conversion
a_values  = np.array(a_values)
z_values  = np.array(z_values)
times_sec = np.array(times)
times_Myr = times_sec / (3.15576e7 * 1e6)
dm_x      = np.array(dm_x)
dm_y      = np.array(dm_y)
dm_z      = np.array(dm_z)
gas_x     = np.array(gas_x)
gas_y     = np.array(gas_y)
gas_z     = np.array(gas_z)

dt_Myr = times_Myr[-1] - times_Myr[0]
print(f"\nSimulation lasts for {dt_Myr} Myr...\n")


# ---- Particle/Gas absolute shift evolution ----
delta_x = dm_x - gas_x                                # step by step absolute shift

# Two panel plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Upper panel: absolute trajectories
ax1.plot(a_values, dm_x, 'o', label='DM Particle (CIC)', markersize=4)
ax1.plot(a_values, gas_x, '-', label='Gas Sphere Center', linewidth=2)
ax1.set_ylabel('Comoving Position X (Mpc)', fontsize=11)
ax1.set_title(f'Absolute shift (z={z_values[0]:.1f} to z={z_values[-1]:.1f})', fontsize=14, fontweight='bold')
ax1.legend()
ax1.tick_params(direction='in', which='both')     
ax1.grid(True, linestyle='--', alpha=0.5)

# Lower panel: DM-gas phase shift
ax2.plot(a_values, delta_x * 1000.0, '-o', markersize=3, label=r'$\Delta X$ (DM - Gas)') # * 1000.0 -> kpc
ax2.axhline(0, color='gray', linestyle=':', alpha=0.8)
ax2.set_xlabel('Scale Factor $a(t)$', fontsize=11)
ax2.set_ylabel(r'Comoving $\Delta X$ (kpc)', fontsize=11)
ax2.legend()
ax2.tick_params(direction='in', which='both')  
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "Absolute_shift.png"), dpi=300)
plt.close()

print(f"\nAbsolute shift plot saved in: {save_path}")


# ---- Shift relative to the particle drift distance ----
dm_drift     = dm_x - dm_x[0]
tot_dm_drift = dm_x[-1] - dm_x[0]
delta_x_rel  = (dm_x - gas_x) / np.abs(tot_dm_drift)   # shift relative to the particle drift distance

print(f"DM particle drifted for {tot_dm_drift:.3f} Mpc during the simuation span (z={z_values[0]:.1f} to z={z_values[-1]:.1f})")

fig_relshift, ax_relshift = plt.subplots()

ax_relshift.set_title(f'Relative shift in {dt_Myr:.1f} Myr', fontsize=12, fontweight='bold', y=1.14)
ax_relshift.plot(dm_drift, delta_x_rel, '-o', markersize=3, label=r'$\Delta X_{\mathrm{rel}}$ (DM - Gas)')
ax_relshift.axhline(0, color='black', linestyle=':', alpha=0.8)
ax_relshift.set_xlabel("Drift distance [Mpc]", fontsize=11)
ax_relshift.set_ylabel(r'$(X_{\mathrm{dm}} - X_{\mathrm{gas}}) / \Delta X_{\mathrm{dm}}^{\mathrm{drift}}$', fontsize=11)
ax_relshift.legend()
ax_relshift.tick_params(direction='in', which='both')  
ax_relshift.grid(True, linestyle='--', alpha=0.5)
ax_relshift.set_xlim(dm_drift[0], dm_drift[-1])

# Creation of a time x axis on the top
ax_top = ax_relshift.twiny()              # create the upper ax
ax_top.set_xlim(ax_relshift.get_xlim())   # align with the lower ax
indices = np.linspace(0, len(z_values) - 1, 5, dtype=int) # only few equispaced indexes

# Map the ticks correspondence
ax_top.set_xticks(dm_drift[indices])
ax_top.set_xticklabels([f"{z_values[idx]:.1f}" for idx in indices])
ax_top.set_xlabel('Redshift $z$', fontsize=11, labelpad=10)
ax_top.tick_params(direction='in')

plt.tight_layout()
plt.savefig(os.path.join(save_path, "Relative_shift.png"), dpi=300)
plt.close()

print(f"\nRelative shift plot saved in: {save_path}")


# ---- Shift evolution in time ----
fig_timeshift, ax_time = plt.subplots(figsize=(7, 5))

ax_time.set_title(f'Shift evolution in time ({tot_dm_drift:.3f} Mpc drift)', fontsize=14, fontweight='bold', y=1.15)
ax_time.plot(times_Myr, delta_x_rel, '-o', markersize=3, label=r'$\Delta X_{\mathrm{rel}}$ (DM - Gas)')
ax_time.axhline(0, color='black', linestyle=':', alpha=0.8)
ax_time.set_xlabel("Cosmic Time [Myr]", fontsize=11)
ax_time.set_ylabel(r'$(X_{\mathrm{dm}} - X_{\mathrm{gas}}) / \Delta X_{\mathrm{dm}}^{\mathrm{drift}}$', fontsize=11)
ax_time.legend(loc='upper left')
ax_time.tick_params(direction='in', which='both')  
ax_time.grid(True, linestyle='--', alpha=0.5)
ax_time.set_xlim(times_Myr[0], times_Myr[-1])

# Creation of a redshift x axis on the top
ax_top = ax_time.twiny()              # create the upper ax
ax_top.set_xlim(ax_time.get_xlim())   # align with the lower ax
tick_times = np.linspace(times_Myr[0], times_Myr[-1], 5)
interpolated_z = interp1d(times_Myr, z_values, kind='linear')(tick_times)

# Map the ticks correspondence
ax_top.set_xticks(tick_times)
ax_top.set_xticklabels([f"{z:.1f}" for z in interpolated_z])
ax_top.set_xlabel('Redshift $z$', fontsize=11, labelpad=10)
ax_top.tick_params(direction='in')

plt.tight_layout()
plt.savefig(os.path.join(save_path, "Time_shift_evolution.png"), dpi=300)
plt.close()

print(f"Time shift evolution plot saved in: {save_path}")




# sistemare la legenda della cic, vedere se togliere frecce vettori velocità

