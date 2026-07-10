#!/usr/bin/env python3
"""
analyze_enzocomp_dmzel.py
======================
Scientific analysis script for compare the DM Zel'dovich Quokka and Enzo tests.

Produces:
  - 

Usage:
    python analyze_enzocomp_dmzel.py --quokka-plotfiles /path/to/output/plt* 
            --enzo-plotfiles /path/to/output/DD* --save /path/to/save
"""

import argparse
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
from unyt import cm, km, s, Mpc, K, g, erg

import h5py
import yt
yt.set_log_level(40)  # suppress yt output except critical errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare DMZeldovich runs from Quokka and Enzo")
    p.add_argument("-q", "--quokka-plotfiles", default=None,
                   help="Glob pattern or directory for Quokka's plotfiles (default: auto-detect)")
    p.add_argument("-e", "--enzo-plotfiles", default=None,
                   help="Glob pattern or directory for Enzo's plotfiles (default: auto-detect)")
    p.add_argument("-s", "--save", default="./outputs",
                   help="Directory to save figures and animations")
    p.add_argument("-a", "--a_collapse", type=float, default=0.5,
                   help="Collapse scale factor (default: 0.5)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
def make_dirs(base):
    subdirs = ["DeltaHistory", "DeltaComparison", "TempComparison" "Animations"]
    for d in subdirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def load_quokka_metadata(quokka_path):
    meta_file = os.path.join(quokka_path, "metadata.yaml")
    if not os.path.exists(meta_file):
        return {}
    with open(meta_file, "r") as f:
        return yaml.safe_load(f)
    
    
def get_cosmo(meta):
    return meta.get("cosmology", {})


def CollectEnzoSnapshots(enzo_dirs):
    """Return sorted Enzo snapshot paths from a list of DD directories."""
    snapshots = []
    for enzo_dir in enzo_dirs:
        snapshots.extend(glob.glob(os.path.join(enzo_dir, "data????")))
    return natsorted(snapshots)


def NormalizeToUnitBox(x, Lbox):
    """Map coordinates from a physical box length to a periodic unit box in [0, 1)."""
    return np.mod(x / Lbox, 1.0)


def AlignPeriodicProfiles(x_ref, y_ref, x_comp, y_comp, L_ref, L_comp):
    """Align two periodic profiles by their density peaks in a common unit box."""
    x_ref_u = NormalizeToUnitBox(x_ref, L_ref)       # normalize postions to unitary box
    x_comp_u = NormalizeToUnitBox(x_comp, L_comp)

    peak_ref = x_ref_u[np.argmax(y_ref)]             # find the index of the y peak
    peak_comp = x_comp_u[np.argmax(y_comp)]
    shift = (peak_ref - peak_comp + 0.5) % 1.0 - 0.5 # +- 0.5 to be in the interval [-0.5, 0.5)

    x_comp_aligned = np.mod(x_comp_u + shift, 1.0)   # let the peaks coincides (according to box periodicity)

    order_ref = np.argsort(x_ref_u)          # rearrange the points accroding to x
    order_comp = np.argsort(x_comp_aligned)
    return x_ref_u[order_ref], y_ref[order_ref], x_comp_aligned[order_comp], y_comp[order_comp]


# ---------------------------------------------------------------------------
# Analytical Zel'dovich solution (Lagrangian grid)
#   x(q) = q - (A(a)/k) * sin(k*q)      [collapse at x=L/2]
#   rho(q) = rho_mean / |1 - A(a)*cos(k*q)|
#   v(q)   = -a * H(a) * (A(a)/k) * sin(k*q)
# ---------------------------------------------------------------------------
def zeldovich_analytical(a, a_collapse, Lx, H0, n_q=2000):
    """Return (x_analytical, delta_analytical, v_analytical) on a Lagrangian grid."""
    k  = 2.0 * np.pi / Lx
    A  = a / a_collapse
    H  = H0 * a**(-1.5)         # EdS: H(a) = H0 * a^{-3/2}

    
    # Consistency with the test: origin at box center    
    q  = np.linspace(0.0, Lx, n_q, endpoint=False)
    q_center = q - 0.5 * Lx                   # set the box origin in the center 0.5*Lx
    displacement = - (A / k) * np.sin(k * q_center)
    x_analytical = q + displacement
    x_analytical = np.mod(x_analytical, Lx)   # wrapper for position perturbed out of domain
    sort_idx = np.argsort(x_analytical)       # check the x sorting after the wrap
    x_analytical = x_analytical[sort_idx]

    delta_analytical = 1.0 / (1.0 - A * np.cos(k * q_center)) - 1.0
    delta_analytical = delta_analytical[sort_idx]

    v_analytical = -a * H * (A / k) * np.sin(k * q_center)
    v_analytical = v_analytical[sort_idx]
    
    return x_analytical, delta_analytical, v_analytical


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    save_path = args.save
    make_dirs(save_path)

    if args.quokka_plotfiles:
        quokka_plotfiles = natsorted(glob.glob(args.quokka_plotfiles))
    else: # try to auto-detect in common output locations
        quokka_candidates = glob.glob("/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_*")
        if not quokka_candidates:
            print("No Quokka's plotfiles found. Use --plotfiles."); sys.exit(1)
        quokka_run_dir = natsorted(quokka_candidates)[-1]   # most recent run
        quokka_plotfiles = natsorted(glob.glob(os.path.join(quokka_run_dir, "plt*")))
        print(f"Auto-detected Quokka's run: {quokka_run_dir}")

    quokka_plotfiles = [p for p in quokka_plotfiles if not p.endswith(".old")]
    len_quokka = len(quokka_plotfiles)

    if not quokka_plotfiles:
        print("No Quokka's plotfiles found!"); sys.exit(1)
    print(f"Found {len_quokka} Quokka's plotfiles.")

    a_collapse = args.a_collapse

    if args.enzo_plotfiles:
        enzo_plotfiles = natsorted(glob.glob(args.enzo_plotfiles))
    else: # try to auto-detect in common output locations
        enzo_candidates = glob.glob("/data/mfulghieri/enzo-dev/outputs/amr_zeldovich")
        if not enzo_candidates:
            print("No Enzo's plotfiles found. Use --plotfiles."); sys.exit(1)
        enzo_run_dir = natsorted(enzo_candidates)[-1]   # most recent run
        enzo_plotfiles = natsorted(glob.glob(os.path.join(enzo_run_dir, "plt*")))
        print(f"Auto-detected Enzo's run: {enzo_run_dir}")

    enzo_plotfiles = [p for p in enzo_plotfiles if not p.endswith(".old")]
    len_enzo = len(enzo_plotfiles)

    if not enzo_plotfiles:
        print("No Enzo's plotfiles found!"); sys.exit(1)
    print(f"Found {len_enzo} Enzo's plotfiles.")

    if len_quokka < len_enzo:
        enzo_plotfiles = enzo_plotfiles[:len_quokka]
        print(f"Matched the number of plotfiles reducing Enzo dataset to {len_quokka} plotfiles: cut the last Enzo's {len_enzo-len_quokka} datasets\n")
    if len_quokka > len_enzo:
        quokka_plotfiles = quokka_plotfiles[:len_enzo]
        print(f"Matched the number of plotfiles reducing Quokka dataset to {len_enzo} plotfiles: cut the last Quokka's {len_quokka-len_enzo} datasets.\n")

    enzo_plt_data_path = CollectEnzoSnapshots(enzo_plotfiles)

    # --- Storage ---
    a_values_quokka = []
    a_values_enzo   = []
    times_quokka    = []
    times_enzo      = []


    frames_delta = []
    frames_temp  = []

    for i, (quokka_plt_path, enzo_plt_path) in enumerate(zip(quokka_plotfiles, enzo_plt_data_path)):
        
        # --- Quokka dataset and metadata ---
        ds_quokka  = yt.load(quokka_plt_path)
        ad_quokka  = ds_quokka.all_data()

        meta = load_quokka_metadata(quokka_plt_path)
        cosmo = get_cosmo(meta)
        a_now_quokka  = cosmo.get("a", float(ds_quokka.current_time))   # fallback to time
        z_now_quokka  = cosmo.get("z", (1.0 / a_now_quokka - 1.0) if a_now_quokka > 0 else 0.0)
        H0_cgs_quokka = cosmo.get("H0", 2.27e-18)                # 70 km/s/Mpc in cgs
        a_values_quokka.append(a_now_quokka)
        times_quokka.append(float(ds_quokka.current_time))
        Lx_quokka = float(ds_quokka.domain_width[0].to("Mpc").v) # physical box size

        # --- Enzo dataset and metadata ---
        ds_enzo = yt.load(enzo_plt_path)
        ad_enzo = ds_enzo.all_data()

        z_now_enzo  = ds_enzo.current_redshift
        a_now_enzo  = 1.0 / (1.0 + z_now_enzo)
        H0_cgs_enzo = ds_enzo.hubble_constant * 100.0 * (1e5 / 3.086e24)
        a_values_enzo.append(a_now_enzo)

        print(f"--- Metadata ---\n")
        print(f"Plotfile {i}: Quokka scale factor: {a_now_quokka:.3f}, Enzo scale factor: {a_now_enzo:.3f} ")
        
        print(ds_quokka.field_list)
        print(ds_enzo.field_list)

        # ---- 1D DM density profile along x for Quokka ----
        c_vals = ds_quokka.domain_center.v
        left_v = ds_quokka.domain_left_edge.v
        right_v = ds_quokka.domain_right_edge.v
        ray_quokka = ds_quokka.ray([left_v[0], c_vals[1], c_vals[2]],
                                   [right_v[0], c_vals[1], c_vals[2]])
        sort_idx_quokka = np.argsort(ray_quokka["index", "x"])
        x_quokka = ray_quokka["index", "x"][sort_idx_quokka].to("Mpc").v
        rho_quokka = ray_quokka["deposit", "CIC_particles_density"][sort_idx_quokka].v
        rho_quokka_mean = np.mean(rho_quokka) if np.mean(rho_quokka) > 0 else 1.0
        delta_quokka = rho_quokka / rho_quokka_mean - 1.0

        # ---- 1D DM density profile along x for Enzo ----
        c_vals_enzo = ds_enzo.domain_center.v
        left_v_enzo = ds_enzo.domain_left_edge.v
        right_v_enzo = ds_enzo.domain_right_edge.v
        ray_enzo = ds_enzo.ray([left_v_enzo[0], c_vals_enzo[1], c_vals_enzo[2]],
                               [right_v_enzo[0], c_vals_enzo[1], c_vals_enzo[2]])
        sort_idx_enzo = np.argsort(ray_enzo["index", "x"])
        x_enzo = ray_enzo["index", "x"][sort_idx_enzo].to("Mpc").v
        rho_enzo = ray_enzo[("enzo", "Dark_Matter_Density")][sort_idx_enzo].v
        rho_enzo_mean = np.mean(rho_enzo) if np.mean(rho_enzo) > 0 else 1.0
        delta_enzo = rho_enzo / rho_enzo_mean - 1.0

        x_quokka_plot, delta_quokka_plot, x_enzo_plot, delta_enzo_plot = AlignPeriodicProfiles(
            x_quokka, delta_quokka, x_enzo, delta_enzo, Lx_quokka, ds_enzo.domain_width[0].to("Mpc").v)

        # ---- Comparison plot ----
        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        ax_comp.plot(x_quokka_plot, delta_quokka_plot, lw=2.0, color="#1f77b4",
                     label=f"Quokka (a={a_now_quokka:.3f})")
        ax_comp.plot(x_enzo_plot, delta_enzo_plot, lw=2.0, color="#d62728",
                     label=f"Enzo (a={a_now_enzo:.3f})")
        ax_comp.set_title(f"Dark matter density contrast | a = {a_now_quokka:.3f} / z = {z_now_quokka:.2f}", fontweight="bold")
        ax_comp.set_xlabel("x / L_box")
        ax_comp.set_ylabel(r"$\delta_{DM} = \rho / \bar{\rho} - 1$")
        ax_comp.set_xlim(0.0, 1.0)
        ax_comp.set_ylim(min(np.min(delta_quokka_plot), np.min(delta_enzo_plot)) * 1.2 - 0.2,
                         max(np.max(delta_quokka_plot), np.max(delta_enzo_plot)) * 1.2 + 0.2)
        ax_comp.grid(True, ls=":", alpha=0.5)
        ax_comp.legend()
        fig_comp.tight_layout()
        fig_comp.savefig(os.path.join(save_path, "DeltaComparison", f"comparison_{i:04d}.png"), dpi=200)
        plt.close(fig_comp)

        frames_delta.append((x_quokka_plot.copy(), delta_quokka_plot.copy(), x_enzo_plot.copy(), delta_enzo_plot.copy(), a_now_quokka, float(ds_quokka.current_time)))

        print(f"[{i + 1:3d}/{min(len(quokka_plotfiles), len(enzo_plt_data_path))}] stored profile for Quokka/Enzo")

    # ---- Animation: overlaid DM density profiles ----
    if len(frames_delta) > 1:
        fig_an, ax_an = plt.subplots(figsize=(10, 5))
        line_quokka, = ax_an.plot([], [], lw=2.2, color="#1f77b4", label="Quokka")
        line_enzo, = ax_an.plot([], [], lw=2.2, color="#d62728", label="Enzo")
        ax_an.set_xlim(0.0, 1.0)
        ax_an.set_xlabel("x / L_box")
        ax_an.set_ylabel(r"$\delta_{DM}$")
        ax_an.grid(True, ls=":", alpha=0.5)
        ax_an.legend(loc="upper right")
        title_an = ax_an.set_title("")

        def update_delta(frame):
            x_q, delta_q, x_e, delta_e, a, time = frames_delta[frame]
            line_quokka.set_data(x_q, delta_q)
            line_enzo.set_data(x_e, delta_e)
            ymin = min(np.min(delta_q), np.min(delta_e)) * 1.2 - 0.2
            ymax = max(np.max(delta_q), np.max(delta_e)) * 1.2 + 0.2
            ax_an.set_ylim(ymin, ymax)
            td_myr = time / (365.25 * 24.0 * 3600.0 * 1e6)
            title_an.set_text(f"Dark matter density contrast | a = {a:.3f}, t = {td_myr:.1f} Myr")
            return line_quokka, line_enzo, title_an

        ani = animation.FuncAnimation(fig_an, update_delta, frames=len(frames_delta), interval=250, blit=False)
        ani_path = os.path.join(save_path, "Animations", "dm_density_comparison.gif")
        ani.save(ani_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_an)
        print(f"Animation saved: {ani_path}")


    # ---- 1D temperature profile along x for Quokka ----
    kb = 1.380649e-16  # erg/K
    mp = 1.672621e-24  # g
    mu = 0.6           # assuming ionized gas
    gamma = 5.0 / 3.0 

    eint_quokka = ray_quokka[('boxlib', 'gasInternalEnergy')][sort_idx_quokka].to('erg/cm**3').v
    rhogas_quokka = ray_quokka[('boxlib', 'gasDensity')][sort_idx_quokka].to('g/cm**3').v
    specen_quokka = eint_quokka / rhogas_quokka
    temp_quokka = ((gamma - 1) * mu * mp) / kb    
    print(eint_quokka)


    # ---- 1D temperature profile along x for Enzo ----
    temp_enzo = ray_enzo[('enzo', 'Temperature')][sort_idx_enzo].to('K').v
    print(temp_enzo)
    

    print("Temperatura Gas Quokka (Valori Centrali):", temp_quokka[len(temp_quokka)//2])
    print("Temperatura Gas Enzo   (Valori Centrali):", temp_enzo[len(temp_enzo)//2])

    # ---- Temperature comparison ----
    fig_temp, ax_temp = plt.subplots(figsize=(10, 5))
    ax_temp.plot(x_quokka_plot, temp_quokka, lw=2.0, color="#1f77b4",
                     label=f"Quokka (a={a_now_quokka:.3f})")
    ax_temp.plot(x_enzo_plot, temp_enzo, lw=2.0, color="#d62728",
                     label=f"Enzo (a={a_now_enzo:.3f})")
    ax_temp.set_title(f"Gas Temperature | a = {a_now_quokka:.3f} / z = {z_now_quokka:.2f}", fontweight="bold")
    ax_temp.set_xlabel("x / L_box")
    ax_temp.set_ylabel(r"$T_{gas} = ((\gamma - 1) * \mu * m_p) / k_b$")
    ax_temp.set_xlim(min(np.min(x_quokka_plot), np.min(x_enzo_plot)) * 1.2 - 0.2,
                         max(np.max(x_quokka_plot), np.max(x_enzo_plot)) * 1.2 + 0.2)
    ax_temp.set_ylim(min(np.min(temp_quokka), np.min(temp_enzo)) * 1.2 - 0.2,
                         max(np.max(temp_quokka), np.max(temp_enzo)) * 1.2 + 0.2)
    ax_temp.grid(True, ls=":", alpha=0.5)
    ax_temp.legend()
    fig_temp.tight_layout()
    fig_temp.savefig(os.path.join(save_path, "TempComparison", f"comparison_{i:04d}.png"), dpi=200)
    plt.close(fig_temp)

    frames_temp.append((x_quokka_plot.copy(), delta_quokka_plot.copy(), x_enzo_plot.copy(), delta_enzo_plot.copy(), a_now_quokka, float(ds_quokka.current_time)))
    
    # ---- Animation: overlaid DM density profiles ----
    if len(frames_temp) > 1:
        fig_an, ax_an = plt.subplots(figsize=(10, 5))
        line_quokka, = ax_an.plot([], [], lw=2.2, color="#1f77b4", label="Quokka")
        line_enzo, = ax_an.plot([], [], lw=2.2, color="#d62728", label="Enzo")
        ax_an.set_xlim(0.0, 1.0)
        ax_an.set_xlabel("x / L_box")
        ax_an.set_ylabel(r"$\delta_{DM}$")
        ax_an.grid(True, ls=":", alpha=0.5)
        ax_an.legend(loc="upper right")
        title_an = ax_an.set_title("")

        def update_temp(frame):
            x_q, temp_q, x_e, temp_e, a, time = frames_temp[frame]
            line_quokka.set_data(x_q, temp_q)
            line_enzo.set_data(x_e, temp_e)
            ymin = min(np.min(temp_q), np.min(temp_e)) * 1.2 - 0.2
            ymax = max(np.max(temp_q), np.max(temp_e)) * 1.2 + 0.2
            ax_an.set_ylim(ymin, ymax)
            td_myr = time / (365.25 * 24.0 * 3600.0 * 1e6)
            title_an.set_text(f"Temperature | a = {a:.3f}, t = {td_myr:.1f} Myr")
            return line_quokka, line_enzo, title_an

        ani = animation.FuncAnimation(fig_an, update_delta, frames=len(frames_temp), interval=250, blit=False)
        ani_path = os.path.join(save_path, "Animations", "dm_temp_comparison.gif")
        ani.save(ani_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_an)
        print(f"Animation saved: {ani_path}")


    print("All done")


if __name__ == "__main__":
    main()



# QUOKKA FIELDS

# [('CIC_particles', 'particle_cpu'), ('CIC_particles', 'particle_id'), 
# ('CIC_particles', 'particle_mass'), ('CIC_particles', 'particle_position_x'),
# ('CIC_particles', 'particle_position_y'), ('CIC_particles', 'particle_position_z'), 
# ('CIC_particles', 'particle_vx'), ('CIC_particles', 'particle_vy'), 
# ('CIC_particles', 'particle_vz'), ('all', 'particle_cpu'), ('all', 'particle_id'), 
# ('all', 'particle_mass'), ('all', 'particle_position_x'), ('all', 'particle_position_y'),
# ('all', 'particle_position_z'), ('all', 'particle_vx'), ('all', 'particle_vy'), 
# ('all', 'particle_vz'), ('boxlib', 'gasDensity'), ('boxlib', 'gasEnergy'), 
# ('boxlib', 'gasInternalEnergy'), ('boxlib', 'gpot'), ('boxlib', 'x-GasMomentum'), 
# ('boxlib', 'y-GasMomentum'), ('boxlib', 'z-GasMomentum'), ('nbody', 'particle_cpu'), 
# ('nbody', 'particle_id'), ('nbody', 'particle_mass'), ('nbody', 'particle_position_x'),
# ('nbody', 'particle_position_y'), ('nbody', 'particle_position_z'),
# ('nbody', 'particle_vx'), ('nbody', 'particle_vy'), ('nbody', 'particle_vz')]


# ENZO FIELDS

# [('enzo', 'Dark_Matter_Density'), ('enzo', 'Density'), ('enzo', 'GasEnergy'), 
# ('enzo', 'Temperature'), ('enzo', 'TotalEnergy'), ('enzo', 'particle_index'), 
# ('enzo', 'particle_mass'), ('enzo', 'particle_position_x'), ('enzo', 'particle_type'), 
# ('enzo', 'particle_velocity_x'), ('enzo', 'x-velocity')]
