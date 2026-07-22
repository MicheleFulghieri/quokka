#!/usr/bin/env python3

import argparse
import glob
import os
import sys
import yaml

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from scipy.integrate import quad
from natsort import natsorted
from unyt import cm, km, s, Mpc, g
from datetime import datetime

import yt
from yt.visualization.volume_rendering.transfer_function_helper import (TransferFunctionHelper,)
yt.set_log_level(40)  # suppress yt output except critical errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Analyze CosmoSphere Quokka run")
    p.add_argument("--plotfiles", default=None,
                   help="Glob pattern or directory for plotfiles (default: auto-detect)")
    p.add_argument("--save", default="./outputs",
                   help="Directory to save figures and animations")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
def make_dirs(base):
    subdirs = ["Hydro", "Animation"]
    for d in subdirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def load_metadata(plt_path):
    meta_file = os.path.join(plt_path, "metadata.yaml")
    if not os.path.exists(meta_file):
        return {}
    with open(meta_file, "r") as f:
        return yaml.safe_load(f)


def get_cosmo(meta):
    return meta.get("cosmology", {})

# ---------------------------------------------------------------------------
# Functions for periodicity management
# ---------------------------------------------------------------------------
# Unwrap gas periodic coordinates (useful for EdS). Spatial coords, in the loops for each plotfiles 
def unwrap_space_coords(gas_xproj, x_array, Lx_box):       
    """ Compute the exact center of mass in a periodic domain using the phase of the first Fourier mode. """
    dx = x_array[1] - x_array[0]
    le = x_array[0] - 0.5 * dx
    theta = 2.0 * np.pi * (x_array - le) / Lx_box  # map coordinates to [0, 2pi] relative to the left edge of the domain
    
    # First Fourier mode components
    xi = np.sum(gas_xproj * np.cos(theta))
    zeta = np.sum(gas_xproj * np.sin(theta))
    
    # The phase directly points to the periodic center of mass
    theta_cm = np.arctan2(zeta, xi)  # arctg2(y,x) account for the relative x and y signs
    gas_center_x = le + (theta_cm / (2.0 * np.pi) * Lx_box) % Lx_box # map the phase back to physical spatial coordinates
    return gas_center_x

# Unwrap DM periodic coordinates (useful for EdS). temporal series, out of the loop
def unwrap_ts_coords(x_array, Lbox):
    """ Unwrap the periodic coordinates in a temporal series into physical coordinates """
    diff = np.diff(x_array, n=1, prepend=x_array[0]) # prepend=x_array[0] to keep the same array dimension
    mask_box_change = (diff < - 0.1 * Lbox)          # boolean mask true (=1) for macroscopic (prevent numerical fluctuations) negative differences
    nbox = np.cumsum(mask_box_change)                # array counter of box switch
    x_unwrapped = x_array + nbox * Lbox
    return x_unwrapped


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    save_path = args.save
    make_dirs(save_path)
   
    if args.plotfiles:
        plotfiles = natsorted(glob.glob(args.plotfiles))
    else:
        # Try to auto-detect in common output locations
        candidates = glob.glob("/data/mfulghieri/quokka/outputs/SphereDMCosmo/SphereDMCosmo_*")
        if not candidates:
            print("No plotfiles found. Use --plotfiles."); sys.exit(1)
        run_dir = natsorted(candidates)[-1]   # most recent run
        plotfiles = natsorted(glob.glob(os.path.join(run_dir, "plt*")))
        print(f"Auto-detected run: {run_dir}")

    plotfiles = [p for p in plotfiles if not p.endswith(".old")]

    if not plotfiles:
        print("No plotfiles found!"); sys.exit(1)
    print(f"Found {len(plotfiles)} plotfiles.")

    # ---- Storage for the history ----
    a_values, z_values, times = [], [], []
    dm_x, dm_y, dm_z          = [], [], []
    dm_vx, dm_vy, dm_vz       = [], [], []
    gas_x, gas_y, gas_z       = [], [], []

    # Storage for animations
    density_frames = [] 
    animation_dir  = os.path.join(save_path, "Animation")
    os.makedirs(animation_dir, exist_ok=True)   # create the folder

    # ---- Initial parameters at snap 0 to initialize the unwrapping functions ----
    ds_init = yt.load(plotfiles[0])  # fist dataset  
    lev_init = 0      # Use level 0 for global center of mass to properly handle AMR without double counting
    refine_factor_init = 2**lev_init
    dims_at_lev_init = ds_init.domain_dimensions * refine_factor_init
    cg_init = ds_init.covering_grid(level=lev_init, left_edge=ds_init.domain_left_edge, dims=dims_at_lev_init)

    Lbox_3d = ds_init.domain_right_edge.to('Mpc').v - ds_init.domain_left_edge.to('Mpc').v
    Lbox_x = Lbox_3d[0]
    Lbox_y = Lbox_3d[1]
    Lbox_z = Lbox_3d[2]

    gdx_init = np.sum(cg_init['boxlib', 'gasDensity'], axis=(1,2))
    dx_init = Lbox_x / ds_init.domain_dimensions[0]
    x_ref_cont = ds_init.domain_left_edge.to('Mpc').v[0] + (np.argmax(gdx_init) + 0.5) * dx_init

    gdy_init = np.sum(cg_init['boxlib', 'gasDensity'], axis=(0,2))
    dy_init = Lbox_y / ds_init.domain_dimensions[1]
    y_ref_cont = ds_init.domain_left_edge.to('Mpc').v[1] + (np.argmax(gdy_init) + 0.5) * dy_init

    gdz_init = np.sum(cg_init['boxlib', 'gasDensity'], axis=(0,1))
    dz_init = Lbox_z / ds_init.domain_dimensions[2]
    z_ref_cont = ds_init.domain_left_edge.to('Mpc').v[2] + (np.argmax(gdz_init) + 0.5) * dz_init

    for i, plt_path in enumerate(plotfiles):
        ds  = yt.load(plt_path)
        ad  = ds.all_data()
        meta = load_metadata(plt_path)
        cosmo = get_cosmo(meta)

        a_init  = cosmo.get('a_init')        
        a_now   = cosmo.get("a", float(ds.current_time))   # fallback to time
        z_now   = cosmo.get("z", (1.0 / a_now - 1.0) if a_now > 0 else 0.0)
        H0_cgs  = cosmo.get("H0", 2.27e-18)                # 70 km/s/Mpc in cgs    
        h       = cosmo.get('hubble_constant')
        omega_m = cosmo.get('Omega_m')
        omega_r = cosmo.get('Omega_r')
        omega_l = cosmo.get('Omega_Lambda')
        omega_k = cosmo.get('Omega_k')
    
        times.append(float(ds.current_time))
        a_values.append(a_now)
        z_values.append(z_now)

        # ---- Tracking positions particle and gas ----
        # Particle
        x_dm_arr = ad['CIC_particles', "particle_position_x"].to('Mpc').v
        y_dm_arr = ad['CIC_particles', "particle_position_y"].to('Mpc').v
        z_dm_arr = ad['CIC_particles', "particle_position_z"].to('Mpc').v
        vx_dm_arr = ad['CIC_particles', "particle_vx"].v
        vy_dm_arr = ad['CIC_particles', "particle_vy"].v
        vz_dm_arr = ad['CIC_particles', "particle_vz"].v
        
        if len(x_dm_arr) > 0:
            dm_x.append(float(x_dm_arr[0]))   # 0 since only 1 particle
            dm_y.append(float(y_dm_arr[0]))
            dm_z.append(float(z_dm_arr[0]))
            dm_vx.append(float(vx_dm_arr[0]))
            dm_vy.append(float(vy_dm_arr[0]))
            dm_vz.append(float(vz_dm_arr[0]))
        
        else:
            dm_x.append(np.nan)            
            dm_y.append(np.nan)
            dm_z.append(np.nan)
            dm_vx.append(np.nan)
            dm_vy.append(np.nan)
            dm_vz.append(np.nan)
        
        # Covering grid creation to extract the density (Level 0 for global center of mass)
        lev = 0 
        refine_factor = 2**lev
        dims_at_lev = ds.domain_dimensions * refine_factor
        cg = ds.covering_grid(level=lev, left_edge=ds.domain_left_edge, dims=dims_at_lev)
        gas_density = cg['boxlib', 'gasDensity']   # shape: [dims_at_lev, dims_at_lev, dims_at_lev]
        # gas_density = ad['boxlib', 'gasDensity'] # old flat extraction for cells [(x0, y0, z0), (x0, y0, z1), ...]
        
        # Grid coordinates extraction
        le_3d = ds.domain_left_edge.to('Mpc').v
        re_3d = ds.domain_right_edge.to('Mpc').v
        Lbox = re_3d - le_3d
        dx = Lbox / dims_at_lev
        x_coordinates = le_3d[0] + (np.arange(dims_at_lev[0]) + 0.5) * dx[0]
        y_coordinates = le_3d[1] + (np.arange(dims_at_lev[1]) + 0.5) * dx[1]
        z_coordinates = le_3d[2] + (np.arange(dims_at_lev[2]) + 0.5) * dx[2]
                
        # Mean position weighted on the density (track sphere center in 3D)
        total_gas_density = np.sum(gas_density)
        if total_gas_density > 0:
            # Density 1D projection
            gdx = np.sum(gas_density, axis=(1,2))  # sum y and z density corresponding to a fixed x
            gdy = np.sum(gas_density, axis=(0,2))
            gdz = np.sum(gas_density, axis=(0,1))
        
            # ---- Unwrap the periodic coordinates (useful for EdS) ----
            gas_center_x = unwrap_space_coords(gdx, x_coordinates, Lbox[0])
            gas_center_y = unwrap_space_coords(gdy, y_coordinates, Lbox[1])
            gas_center_z = unwrap_space_coords(gdz, z_coordinates, Lbox[2])
        
            gas_x.append(float(gas_center_x))
            gas_y.append(float(gas_center_y))
            gas_z.append(float(gas_center_z))
        else:
            gas_x.append(np.nan)
            gas_y.append(np.nan)
            gas_z.append(np.nan)


        # ---- Hydro analysis ----
        
        # Single SlicePlot for all fields via list of fields: yt reads the disk only once per file
        fields = [('boxlib', 'gasDensity'), ('boxlib', 'gasInternalEnergy'), ('boxlib', 'x-GasMomentum')]
        slc = yt.SlicePlot(ds, 'z', fields, center='c')   
        
        # Global annotations valid for all plots
        slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
        slc.annotate_scale(corner="upper_right")
        slc.annotate_grids(alpha=0.3, min_level=1)  # visualize AMR refined grids
        slc.annotate_particles(width=(0.5, 'Mpc'), p_size=40, col='white', marker='o', ptype='CIC_particles')
        
        # Specific settings for each field
        slc.set_cmap(('boxlib', 'gasDensity'), cmap="Blues")
        slc.set_zlim(('boxlib', 'gasDensity'), 1e-31, 1e-20)  # colorbar CGS range
        
        slc.set_cmap(('boxlib', 'gasInternalEnergy'), 'dusk')
        #slc.set_zlim(('boxlib', 'gasInternalEnergy'), 1e-29, 1e-16)
        
        slc.set_cmap(('boxlib', 'x-GasMomentum'), 'dusk')
        
        # Force matplotlib rendering to manipulate axes and save custom figures
        slc._setup_plots()
        
        # Matplotlib logic to insert the particle legend box only on the Density plot
        ax = slc.plots[('boxlib', 'gasDensity')].axes
        text_legend = r"$\circ$ DM Particle (CIC)"
        legend_box = AnchoredText(text_legend, loc='lower left', pad=0.7, borderpad=0.7,
                                       prop=dict(size=13, color='black'), frameon=True)
        legend_box.patch.set_facecolor('white')
        legend_box.patch.set_alpha(0.95)
        legend_box.patch.set_edgecolor('gray')
        ax.add_artist(legend_box)
        
        # Create folders (if they do not exist) and save
        os.makedirs(os.path.join(save_path, "Hydro", "Density"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "Hydro", "Eint"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "Hydro", "XMomentum"), exist_ok=True)
        
        dens_path = os.path.join(save_path, "Hydro", "Density", f"Density_{i:03d}.png")
        eint_path = os.path.join(save_path, "Hydro", "Eint", f"InternalEnergy_{i:03d}.png")
        mom_path  = os.path.join(save_path, "Hydro", "XMomentum", f"XMomentum_{i:03d}.png")
        
        slc.plots[('boxlib', 'gasDensity')].figure.savefig(dens_path, dpi=300, bbox_inches='tight')
        density_frames.append(dens_path)
        
        slc.plots[('boxlib', 'gasInternalEnergy')].figure.savefig(eint_path, dpi=300, bbox_inches='tight')
        slc.plots[('boxlib', 'x-GasMomentum')].figure.savefig(mom_path, dpi=300, bbox_inches='tight')
        
        # y-Momentum
        # slc = yt.SlicePlot(ds, 'z', ('boxlib', 'y-GasMomentum'))
        # slc.set_cmap(('boxlib', 'y-GasMomentum'), 'inferno')
        # #slc.annotate_quiver(('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), factor=12)
        # slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
        # slc.annotate_scale(corner="upper_right")
        # slc.save(os.path.join(save_path + "/Hydro/YMomentum", f"YMomentum_{i:03d}.png"))
        
        ds.index.clear_all_data()  # free the RAM

        if (i == len(plotfiles) - 1): 
            # Extract grid and domain parameters from the last analyzed dataset object (ds)
            domain_dimensions = ds.domain_dimensions[0]                # Grid resolution (Nx) assuming cubic domain
            box_length_mpc = float(ds.domain_width.to('Mpc')[0])       # Box side length in Mpc
            cell_size_mpc = box_length_mpc / domain_dimensions         # Size of a single cell in Mpc
            
            print("\n" + "="*58)
            print("                    DOMAIN SUMMARY")
            print("="*58)
            print(f" Domain resolution (Cells)       : {domain_dimensions} x {domain_dimensions} x {domain_dimensions}")
            print(f" Physical box size length        : {box_length_mpc:.2f} Mpc")
            print(f" Single cell grid size           : {cell_size_mpc:.4f} Mpc ({cell_size_mpc*1000.0:.2f} kpc)")
            
            
            # Numpy array conversion
            a_values  = np.array(a_values)
            z_values  = np.array(z_values)
            times_sec = np.array(times)
            times_Myr = times_sec / (3.15576e7 * 1e6)
            dm_x      = np.array(dm_x)
            dm_y      = np.array(dm_y)
            dm_z      = np.array(dm_z)
            dm_vx     = np.array(dm_vx)
            dm_vy     = np.array(dm_vy)
            dm_vz     = np.array(dm_vz)
            gas_x     = np.array(gas_x)
            gas_y     = np.array(gas_y)
            gas_z     = np.array(gas_z)
            
            dt_Myr = times_Myr[-1] - times_Myr[0]
            print(f"\n Initial velocity of gas and DM (Vx, Vy, Vz): ({dm_vx[0]:.2e}, {dm_vy[0]:.2e}, {dm_vz[0]:.2e}) cm/s")
            print(f"\nSimulation lasts for {dt_Myr} Myr...\n")
                
            # ---- Unwrap the periodic coordinates (useful for EdS) ----
            box_length_mpc = float(ds.domain_width.to('Mpc')[0])   # box side length in Mpc
            dm_x  = unwrap_ts_coords(dm_x, box_length_mpc)
            dm_y  = unwrap_ts_coords(dm_y, box_length_mpc)
            dm_z  = unwrap_ts_coords(dm_z, box_length_mpc)
            gas_x = unwrap_ts_coords(gas_x, box_length_mpc)
            gas_y = unwrap_ts_coords(gas_y, box_length_mpc)
            gas_z = unwrap_ts_coords(gas_z, box_length_mpc)
            
            for i in range(len(gas_x)):
                print(f"  snap {i:03d} | gas_x = {gas_x[i]:.6f} | dm_x = {dm_x[i]:.6f} | delta = {(dm_x[i]-gas_x[i])*1000:.3f} Mpc")
            
            
            # ---- Particle/Gas absolute shift evolution ----
            delta_x = dm_x - gas_x                                # step by step absolute shift
            
            # Two panel plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Upper panel: absolute trajectories
            ax1.plot(a_values, dm_x, 'o', label='DM Particle (CIC)', markersize=4)
            ax1.plot(a_values, gas_x, '-', label='Gas Sphere Center', linewidth=2)
            ax1.set_ylabel('Comoving Position X (Mpc)', fontsize=11)
            ax1.set_xscale('log')
            ax1.set_title(f'Absolute shift (z={z_values[0]:.1f} to z={z_values[-1]:.1f})', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.tick_params(direction='in', which='both')     
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Lower panel: DM-gas phase shift
            ax2.plot(a_values, delta_x * 1000.0, '-o', markersize=3, label=r'$\Delta X$ (DM - Gas)') # * 1000.0 -> kpc
            ax2.axhline(0, color='gray', linestyle=':', alpha=0.8)
            ax2.set_xscale('log')
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
            
            print(f"DM particle drifted for {tot_dm_drift:.3f} cMpc during the simulation span (z={z_values[0]:.1f} to z={z_values[-1]:.1f})")
            
            fig_relshift, ax_relshift = plt.subplots()
            
            ax_relshift.set_title(f'Relative shift $\cdot 10^{7}$ in {dt_Myr:.1f} Myr', fontsize=12, fontweight='bold', y=1.14)
            ax_relshift.plot(dm_drift, delta_x_rel * 10**7, '-o', markersize=3, label=r'$\Delta X_{\mathrm{rel}}$ (DM - Gas) $\times 10^7$')
            ax_relshift.axhline(0, color='black', linestyle=':', alpha=0.8)
            ax_relshift.set_xlabel("Drift distance [cMpc]", fontsize=11)
            ax_relshift.set_ylabel(r'$\left[ (X_{\mathrm{dm}} - X_{\mathrm{gas}}) / \Delta X_{\mathrm{dm}}^{\mathrm{drift}} \right] \cdot 10^{7}$', fontsize=11)
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
            
            ax_time.set_title(fr'Shift evolution in time ({tot_dm_drift:.3f} Mpc drift ($\cdot 10^{7}$))', fontsize=14, fontweight='bold', y=1.15)
            ax_time.plot(times_Myr, delta_x_rel * 10**7, '-o', markersize=3, label=r'$\Delta X_{\mathrm{rel}}$ (DM - Gas) $\times 10^7$')  # 10**7 to avoid overlap with the twin x upper ax of the exponent
            ax_time.axhline(0, color='black', linestyle=':', alpha=0.8)
            ax_time.set_xscale('log')
            ax_time.set_xlabel("Cosmic Time [Myr]", fontsize=11)
            ax_time.set_ylabel(r'$\left[ (X_{\mathrm{dm}} - X_{\mathrm{gas}}) / \Delta X_{\mathrm{dm}}^{\mathrm{drift}} \right] \cdot 10^{7}$', fontsize=11)
            ax_time.legend(loc='best')
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
            
            
            # ----  DM velocity profile ----
            
            fig_vel, ax_vel = plt.subplots()
            
            ax_vel.set_title(f'Particle velocity profile {dt_Myr:.1f} Myr', fontsize=12, fontweight='bold', y=1.14)
            ax_vel.plot(times_Myr, dm_vx, '-o', markersize=3, label=r'$v_x^{\mathrm{DM}}$ [cm/s]')
            ax_vel.axhline(dm_vx[0], color='black', linestyle=':', alpha=0.8, label='Initial DM velocity')
            ax_vel.set_xscale('log')
            ax_vel.set_xlabel("Cosmic Time [Myr]", fontsize=11)
            ax_vel.set_ylabel(r'$v_x^{\mathrm{DM}}$ [cm/s]', fontsize=11)
            ax_vel.legend()
            ax_vel.tick_params(direction='in', which='both')  
            ax_vel.grid(True, linestyle='--', alpha=0.5)
            ax_vel.set_xlim(times_Myr[0], times_Myr[-1])
            
            # Creation of a redshift x axis on the top
            ax_top_vel = ax_vel.twiny()              # create the upper ax
            ax_top_vel.set_xlim(ax_vel.get_xlim())   # align with the lower ax
            tick_times = np.linspace(times_Myr[0], times_Myr[-1], 5)
            interpolated_z = interp1d(times_Myr, z_values, kind='linear')(tick_times)
            
            # Map the ticks correspondence
            ax_top_vel.set_xticks(tick_times)
            ax_top_vel.set_xticklabels([f"{z:.1f}" for z in interpolated_z])
            ax_top_vel.set_xlabel('Redshift $z$', fontsize=11, labelpad=10)
            ax_top_vel.tick_params(direction='in')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "DM_xvel_evolution.png"), dpi=300)
            plt.close()
            
            print(f"DM velocity evolution plot saved in: {save_path}")
            
            # Analyical solution DM vx validation and norms
            dm_analytical_vel = dm_vx[0] * (a_values[0] / a_values)  # Hubble drag
            
            print(f"\n DM x-velocity analysis:")
            print(f"  Initial Velocity: {dm_vx[0]:.2e} cm/s")
            print(f"  Final Velocity:   {dm_vx[-1]:.2e} cm/s")
            
            vel_rel_error = (dm_vx- dm_analytical_vel) / dm_analytical_vel
            
            # Velocity norms
            L1_vel   = np.mean(np.abs(vel_rel_error))         # arithmetic mean of errors
            L2_vel   = np.sqrt(np.mean(vel_rel_error**2))     # square mean
            Linf_vel = np.max(vel_rel_error)                  # worst relative error
            
            print(f"\nGlobal error norms (peculiar velocities):")
            print(f"  L1   = {L1_vel:.6e}")
            print(f"  L2   = {L2_vel:.6e}")
            print(f"  Linf = {Linf_vel:.6e}")
            
            
            fig_vel_val, ax_vel_val = plt.subplots()
            ax_vel_val.plot(times_Myr, dm_vx, 'o', label="Numerical DM velocity")
            ax_vel_val.plot(times_Myr, dm_analytical_vel, label=r"Analytical solution: $v_0 \cdot \frac{a_{\mathrm{in}}}{a}$")
            ax_vel_val.set_title("Peculiar Velocity Decay: Hubble Drag", pad=25, fontsize=14, fontweight='bold')
            ax_vel_val.set_xscale('log')
            ax_vel_val.set_xlabel("Cosmic Time [Myr]", fontsize=11)
            ax_vel_val.set_ylabel(r'$v_x^{\mathrm{DM}}$ [cm/s]', fontsize=11)
            ax_vel_val.legend()
            ax_vel_val.tick_params(direction='in', which='both')  
            ax_vel_val.grid(True, linestyle='--', alpha=0.5)
            ax_vel_val.set_xlim(times_Myr[0], times_Myr[-1])
            
            # Creation of a redshift x axis on the top
            ax_top_vel_val = ax_vel_val.twiny()              # create the upper ax
            ax_top_vel_val.set_xlim(ax_vel_val.get_xlim())   # align with the lower ax
            tick_vel_val = np.linspace(times_Myr[0], times_Myr[-1], 5)
            interpolated_z = interp1d(times_Myr, z_values, kind='linear')(tick_times)
            
            # Map the ticks correspondence
            ax_top_vel_val.set_xticks(tick_vel_val)
            ax_top_vel_val.set_xticklabels([f"{z:.1f}" for z in interpolated_z])
            ax_top_vel_val.set_xlabel('Redshift $z$', fontsize=11, labelpad=10)
            ax_top_vel_val.tick_params(direction='in')
            
            # # Text insertion in the figure
            # stats_text_mom = (f'L1 Error: {L1_vel:.2e}\n'
            #                   f'L2 Error: {L2_vel:.2e}\n'
            #                   f'Linf Error: {Linf_vel:.2e}')
            # ax_vel_val.text(0.65, 0.70, stats_text_mom,
            #         transform=ax.transAxes, fontsize=10, verticalalignment='top',
            #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "DM_xvel_validation.png"), dpi=300)
            plt.close()
            
            print(f"DM velocity validation plot saved in: {save_path}")
            
            # Fit analysis
            log_a     = np.log10(a_values)
            log_DM_vx = np.log10(dm_vx)
            
            # Linear least squares polynomial fit
            slope, intercept = np.polyfit(log_a, log_DM_vx, 1)   # 1: linear, log(p) = intercept + slope * log(a)
            
            print(f"Velocity power law analysis:")
            print(f"Theoretical exponent (Hubble drag only): -1.0") 
            print(f"Fit of Quokka exponent: {slope:.4f}")
            print(f"Difference: {np.abs(-1.0 - slope):.2e}")
            
            
            # ---- Simulation summary ----
            
            # Extract grid and domain parameters from the last analyzed dataset object (ds)
            domain_dimensions = ds.domain_dimensions[0]                # Comoving grid resolution (Nx) assuming cubic domain
            box_length_mpc = float(ds.domain_width.to('Mpc')[0])       # Comoving box side length in Mpc
            cell_size_mpc = box_length_mpc / domain_dimensions         # COmoving size of a single cell in Mpc
            final_shift_cells = delta_x[-1] / cell_size_mpc            # Final DM-Gas misalignment in cells
            
            # Analytical solution (for EdS only) 
            # phys_dm_drift =  2 * (dm_vx[0] * a_values[0] / H0_cgs) * (np.sqrt(a_values[-1]) - np.sqrt(a_values[0]))
            # print(f" Total physical distance traveled by DM  {phys_dm_drift :.4f} Mpc")
            
            
            print("\n" + "="*58)
            print("                    FINAL SIMULATION SUMMARY")
            print("="*58)
            print(f" Simulation spans from a = {a_values[0]:.2e} (z = {z_values[0]:.1f}) to a = {a_values[-1]:.2e} (z = {z_values[-1]:.1f})")
            print(f" Initial drift velocity (DM/Gas) : {dm_vx[0]:.2e} cm/s ({dm_vx[0]/1e5:.1f} km/s)")
            print(f" Total simulation time duration  : {dt_Myr:.2f} Myr")
            print(f" Total distance traveled by DM   : {tot_dm_drift:.4f} Mpc")
            print(f" Final shift (DM - Gas)          : {delta_x[-1]*1000.0:.7f} kpc ({final_shift_cells:.7f} cells)")
            print("-"*58)
            
            
            if len(density_frames) > 1:
                import matplotlib.animation as animation
                from PIL import Image
            
                print("Generating hydro density evolution GIF (with annotated DM particle)...")
            
                # Square figure for the gif
                fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
                ax_anim.set_xticks([]) 
                ax_anim.set_yticks([])
                ax_anim.axis('off')   

                # Load first image to initialize the object 
                first_png_path = density_frames[0]
                img_data = np.array(Image.open(first_png_path))      # take the first image
                img_obj = ax_anim.imshow(img_data)                   # convert the save paths 
                title_ha = ax_anim.set_title("")
                fig_anim.tight_layout()
            
                def update_density_slice(frame):  # frame is integer increasing at every step
                    img_path = density_frames[frame]
                    img_data = np.array(Image.open(img_path)) # read the data
                    img_obj.set_data(img_data)                # substitute the pixels
                    img_obj.set_data(img_data)                # update screen pixels
                    title_ha.set_text(f"Gas Density")
                    return [img_obj, title_ha]
            
                # Animation engine
                ani_slice = animation.FuncAnimation(fig_anim, update_density_slice,
                                                frames=len(density_frames), interval=250, blit=True)
            
                # Save the gif 
                ani_slice_path = os.path.join(animation_dir, "gas_density_evolution.gif")
                ani_slice.save(ani_slice_path, writer="pillow", fps=4, dpi=150)
                plt.close(fig_anim)
            
            print(f"-> Animation successfully saved at: {ani_slice_path}\n")


if __name__ == "__main__":
    main()


# python3 /data/mfulghieri/quokka/my_worksite/myAnalysis/SphereDMCosmo/analyze_sphereDM.py \
#        --plotfiles "/data/mfulghieri/quokka/outputs/SphereDMCosmo/SphereDMCosmo_13403_EdS_R60kpc/plt*" \
#        --save "/data/mfulghieri/quokka/my_worksite/myAnalysis/SphereDMCosmo/outputs"


        
        
        


