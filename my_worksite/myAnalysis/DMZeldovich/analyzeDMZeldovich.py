#!/usr/bin/env python3
"""
analyzeDMZeldovich.py
======================
Scientific analysis script for the DM Zel'dovich pancake test (Quokka).

Produces:
  - History of DM density contrast evolution (all snapshots on one figure)
  - Per-snapshot comparison: numerical vs analytical Zel'dovich profile
  - L2 error between numerical and analytical δ(x) profile, as a function of a
  - Per-snapshot phase-space diagram (vx vs x)
  - 2D DM density histogram (x-y plane, normalized to particles/kpc^2, LogNorm)
  - 3D particle scatter (one per snapshot, azimuth-rotating animation)
  - MP4/GIF animation of density profile evolution
  - MP4/GIF animation of phase-space evolution
  - Summary panel (4-panel figure)

Usage:
    python analyzeDMZeldovich.py --plotfiles /path/to/output/plt* --save /path/to/save
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
from matplotlib.colors import Normalize, LogNorm
from natsort import natsorted
from unyt import cm, s
from datetime import datetime
import pprint

import yt
yt.set_log_level(40)  # suppress yt output except critical errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Analyze DMZeldovich Quokka run")
    p.add_argument("--plotfiles", default=None,
                   help="Glob pattern or directory for plotfiles (default: auto-detect)")
    p.add_argument("--save", default="./outputs",
                   help="Directory to save figures and animations")
    p.add_argument("--a_collapse", type=float, default=0.5,
                   help="Collapse scale factor (default: 0.5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
def make_dirs(base):
    subdirs = ["DeltaHistory", "DeltaComparison", "2D_DM_Density", "PhaseSpace",
               "ParticlePositions", "Animations"]
    for d in subdirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def load_metadata(plt_path):
    meta_file = os.path.join(plt_path, "metadata.yaml")
    if not os.path.exists(meta_file):
        return {}                   # fallback: empy dictionary
    with open(meta_file, "r") as f:
        return yaml.safe_load(f)    # convert the yaml in a python dictionary


def get_cosmo(meta):
    return meta.get("cosmology", {})


def read_a_from_ds(ds, meta):
    """
    Read the comoving scale factor a from the AMReX plotfile header.
    AMReX writes it as 'comoving_a' in ds.parameters (the Header file).
    Falls back to the yaml metadata, then to ds.current_time as last resort.
    """
    # Primary: AMReX plotfile header attribute
    if "a" in ds.parameters:
        return float(ds.parameters["a"])
    # Secondary: yaml metadata
    cosmo = get_cosmo(meta)
    if "a" in cosmo:
        return float(cosmo["a"])
    # Fallback: current_time (only valid if a == t for EdS normalisation)
    print("Warning: Using ds.current_time as a fallback for the scale factor. It was not found in the plotfile header or metadata.")    
    return float(ds.current_time)


# ---------------------------------------------------------------------------
# Analytical Zel'dovich solution (Lagrangian grid)
#   x(q) = q - (A(a)/k) * sin(k*(q - L/2))   [collapse centred at box centre]
#   rho(q) = rho_mean / |1 - A(a)*cos(k*(q-L/2))|
#   delta(q) = rho/rho_mean - 1
#   v(q)   = -a * H(a) * (A(a)/k) * sin(k*(q-L/2))
#
# The peak of the profile is at q_center=0 -> cos(0)=1:
#   delta_max = 1/(1 - A) - 1 = A/(1-A) = (a/a_c) / (1 - a/a_c)
# ---------------------------------------------------------------------------
def zeldovich_analytical(a, a_collapse, Lx, rho_mean, H0=None, G_CGS = 6.67430e-8, n_q=2000):
    """Return (x_analytical, delta_analytical, v_analytical) on a Lagrangian grid."""
    k  = 2.0 * np.pi / Lx
    A  = a / a_collapse

    # Use the effective Hubble parameter derived from simulation mean density if
    # available, otherwise fall back to the provided H0.
    if H0 is None:
        # Derive H0 from rho_mean using EdS critical density: rho_mean = 3 H0^2 / (8 pi G)
        H0 = np.sqrt(rho_mean * 8.0 * np.pi * G_CGS / 3.0)
    H  = H0 * a**(-1.5)         # EdS: H(a) = H0 * a^{-3/2}

    # Consistency with the test initialisation
    q  = np.linspace(0.0, Lx, n_q, endpoint=False)
    q_center = q - 0.5 * Lx                   # shift origin to box centre
    displacement = - (A / k) * np.sin(k * q_center)
    x_analytical = q + displacement
    x_analytical = np.mod(x_analytical, Lx)   # periodic wrap
    sort_idx = np.argsort(x_analytical)       # re-sort after wrap
    x_analytical = x_analytical[sort_idx]

    delta_analytical = 1.0 / (1.0 - A * np.cos(k * q_center)) - 1.0
    delta_analytical = delta_analytical[sort_idx]

    v_analytical = -a * H * (A / k) * np.sin(k * q_center)
    v_analytical = v_analytical[sort_idx]

    return x_analytical, delta_analytical, v_analytical


# ---------------------------------------------------------------------------
# L2 error between numerical and analytical delta(x) profile
# ---------------------------------------------------------------------------
def profile_l2_error(x_num, delta_num, x_anal, delta_anal):
    """
    Interpolate the analytical profile onto the numerical x-grid and compute
    the root-mean-square difference.
    Returns a single scalar (dimensionless, same units as delta).
    """
    delta_anal_interp = np.interp(x_num, x_anal, delta_anal)
    return float(np.sqrt(np.mean((delta_num - delta_anal_interp)**2)))


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
        candidates = glob.glob("/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_*")
        if not candidates:
            print("No plotfiles found. Use --plotfiles."); sys.exit(1)
        run_dir = natsorted(candidates)[-1]   # most recent run
        plotfiles = natsorted(glob.glob(os.path.join(run_dir, "plt*")))
        print(f"Auto-detected run: {run_dir}")

    plotfiles = [p for p in plotfiles if not p.endswith(".old")]  #[:1]

    if not plotfiles:
        print("No plotfiles found!"); sys.exit(1)
    print(f"Found {len(plotfiles)} plotfiles.")

    a_collapse = args.a_collapse

    # --- Storage for history ---
    a_values        = []
    times           = []
    delta_max_vals  = []
    l2_errors       = []    
    x_coords_hist   = []
    delta_hist      = []

    # Figures that accumulate across snapshots
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("coolwarm_r")

    # Storage for animations
    frames_delta  = []   # (x_coord, delta, x_anal, delta_anal, a_val, current_time)
    frames_2d     = []   # (density_map_phys, xedges, yedges, a_now, current_time)
    frames_phase  = []   # (px, pvx, a_val)
    frames_3d     = []   # (px, py, pz, vx, a_now, current_time)

    for i, plt_path in enumerate(plotfiles):
        ds  = yt.load(plt_path)
        ad  = ds.all_data()
        meta = load_metadata(plt_path)

        cm_to_kpc = 1.0 / 3.08567758e21

        # Gravitational costant from metadata
        constants = meta.get("constants", {})
        if "G" in constants:
            G_CGS = constants.get("G")
        else:
            G_CGS = 6.67430e-8  # cm^3 g^-1 s^-2
            print(f"No G costant found in metadata: using fallback value {G_CGS} cm^3 g^-1 s^-2")

        # Cosmological metadata
        cosmo = get_cosmo(meta)  # funtion for meta.get()
        if(i==0):
            print("=" * 60)
            print("  COSMOLOGICAL PARAMETERS (Snapshot 0)")
            print("=" * 60)            
            pprint.pprint(meta, indent=4, width=80, compact=False)
            pprint.pprint(cosmo, indent=4, width=80, compact=False)
            print("\n" + "=" * 60 + "\n")

        # Read a(t) from the AMReX plotfile header (most reliable source)
        a_now  = read_a_from_ds(ds, meta)
        z_now  = (1.0 / a_now - 1.0) if a_now > 0 else 0.0

        a_values.append(a_now)
        times.append(float(ds.current_time))

        # Physical box sizes [kpc] and [cm]
        Lx = float(ds.domain_width[0].to("kpc").v)
        Ly = float(ds.domain_width[1].to("kpc").v)
        Lz = float(ds.domain_width[2].to("kpc").v)
        Lx_cm = float(ds.domain_width[0].to("cm").v)
        Ly_cm = float(ds.domain_width[1].to("cm").v)
        Lz_cm = float(ds.domain_width[2].to("cm").v)

        ncells   = ds.domain_dimensions
        if(i==0):
            print(f"Simulation cell resolution: {ncells} cells")


        # ---- 1D DM density profile via Planar Average (Y-Z projection) ----
        # Covering grid, potentially optimized for AMR (by changing lev)
        # lev = 0  
        # refine_factor = 2**lev
        # dims_at_lev = ds.domain_dimensions * refine_factor
        # cg = ds.covering_grid(level=lev, left_edge=ds.domain_left_edge, dims=dims_at_lev)

        # rho_3d = cg["deposit", "CIC_particles_density"].v  # shape: [dims_at_lev, dims_at_lev, dims_at_lev]
        # rho_dm = np.mean(rho_3d, axis=(1,2))               # project along x: shape [dims_at_lev]
        # dx = Lx / dims_at_lev[0]
        # x_coord = np.linspace(0.5 * dx, Lx - 0.5 * dx, dims_at_lev[0])   # accounting for cell centered
        # rho_dm_mean = np.mean(rho_dm) if np.mean(rho_dm) > 0 else 1.0
        # delta     = rho_dm / rho_dm_mean - 1.0
        # delta_max = float(np.max(delta))

        
        # # ---- 1D DM density profile along x (ray through box centre) ----
        c_vals  = ds.domain_center.v
        left_v  = ds.domain_left_edge.v
        right_v = ds.domain_right_edge.v
        ray = ds.ray([left_v[0], c_vals[1], c_vals[2]],
                     [right_v[0], c_vals[1], c_vals[2]])
        sort_idx  = np.argsort(ray["index", "x"])
        x_coord   = ray["index", "x"][sort_idx].to("kpc").v
        rho_dm    = ray["deposit", "CIC_particles_density"][sort_idx].v
        rho_dm_mean = np.mean(rho_dm) if np.mean(rho_dm) > 0 else 1.0
        delta     = rho_dm / rho_dm_mean - 1.0
        delta_max = float(np.max(delta))
        print(f"Sampling the density 1D LoS: number of bins along the skewer: {len(rho_dm)}, given by len(rho_dm)")

        delta_max_vals.append(delta_max)
        x_coords_hist.append(x_coord)
        delta_hist.append(delta)

        color_val = cmap(i / max(len(plotfiles) - 1, 1))

        # History plot (accumulate)
        ax_hist.plot(x_coord, delta, color=color_val,
                     lw=1.2, label=f"a={a_now:.3f}")

        # ---- Analytical solution (full profile) ----
        if "comoving_mean_density" in ds.parameters: 
            rho_mean = float(ds.parameters["comoving_mean_density"])
            info_source = "header comoving_mean_density"
        elif "comoving_mean_density" in cosmo:       
            rho_mean = float(cosmo["comoving_mean_density"])
            info_source = "yaml  metadata comoving_mean_density"
        else:
            rho_mean = float(np.mean(rho_dm)) if np.mean(rho_dm) > 0 else 1.0
            info_source = "numerically computed mean density"

        x_anal_cm, delta_anal, v_anal_cgs = zeldovich_analytical(
            a_now,      # analytical reconstruction of the gravitational history that Quokka simulated having only a_init as an input
            a_collapse,
            Lx_cm,
            rho_mean=rho_mean,
            G_CGS=G_CGS,
            H0=None)
        x_anal = x_anal_cm * cm_to_kpc
        if(i==0):
            print(f"Using {info_source} = {rho_mean:.4e} g/cm^3 for analytical H0 derivation")

        # ---- L2 error: numerical vs analytical δ(x) profile ----
        l2 = profile_l2_error(x_coord, delta, x_anal, delta_anal)
        l2_errors.append(l2)
        print(f" Profile L2 error = {l2:.3f}")

        # ---- Per-snapshot comparison: numerical vs analytical profile ----
        fig_c, ax_c = plt.subplots(figsize=(10, 5))
        ax_c.plot(x_coord, delta, lw=2, color="#1f77b4", label="Quokka (numerical)")
        ax_c.plot(x_anal, delta_anal, lw=2, ls="--", color="crimson", label="Zel'dovich (analytical)")
        ax_c.set_title( f"DM density contrast  |  $a = {a_now:.3f}$ ($z = {z_now:.2f}$)" , fontweight="bold")
        ax_c.set_xlabel("x [kpc]")
        ax_c.set_ylabel(r"$\delta = \rho/\bar\rho - 1$")
        ax_c.set_xlim(0, Lx)
        ax_c.set_ylim(max(-1.5, min(delta) * 1.3), min(delta_max * 2.0 + 0.5, 50))
        ax_c.legend()
        ax_c.grid(True, ls=":", alpha=0.5)
        fig_c.tight_layout()
        fig_c.savefig(os.path.join(save_path, "DeltaComparison", f"comparison_{i:04d}.png"), dpi=200)
        plt.close(fig_c)

        # Store for animation
        frames_delta.append((x_coord.copy(), delta.copy(),
                              x_anal.copy(), delta_anal.copy(),
                              a_now, float(ds.current_time)))

        # ---- 2D Histogram of Particle Density (x-y plane) ----
        # Normalised to surface density [particles/kpc^2] 
        # LogNorm is applied automatically when the dynamic range exceeds 10×.
        px_2d = ad["CIC_particles", "particle_position_x"].to("kpc").v
        py_2d = ad["CIC_particles", "particle_position_y"].to("kpc").v
        bins = ncells[0]
        dA   = (Lx / bins) * (Ly / bins)   # pixel area [kpc^2]

        density_map, xedges, yedges = np.histogram2d(
            px_2d, py_2d,
            bins=bins,
            range=[[0, Lx], [0, Ly]])
        density_map_phys = density_map / dA  # [particles / kpc^2]

        # Colour scale
        vmin_map = max(density_map_phys[density_map_phys > 0].min(), 1e-3)
        vmax_map = density_map_phys.max()
        norm_map = (LogNorm(vmin=vmin_map, vmax=vmax_map)
                    if vmax_map / vmin_map > 10
                    else Normalize(vmin=vmin_map, vmax=vmax_map))

        fig2d, ax2d = plt.subplots(figsize=(6, 5))
        im = ax2d.imshow(density_map_phys.T, origin="lower",
                         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         cmap="viridis", norm=norm_map)
        ax2d.set_title(f"DM Surface Density ($a = {a_now:.3f}$)",
                       fontsize=12, fontweight="bold")
        ax2d.set_xlabel("X [kpc]", fontsize=11)
        ax2d.set_ylabel("Y [kpc]", fontsize=11)
        cbar = fig2d.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Sigma_\mathrm{DM}$ [particles kpc$^{-2}$]", fontsize=10)
        fig2d.savefig(os.path.join(save_path, "2D_DM_Density",
                                   f"Density2D_{i:03d}.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig2d)

        # Store for animation (store physical map and edges)
        frames_2d.append((density_map_phys.copy(), xedges.copy(), yedges.copy(),
                          a_now, float(ds.current_time)))

        # ---- Phase space (vx vs x) ----
        px  = ad["CIC_particles", "particle_position_x"].to("kpc").v
        pvx_raw = ad["CIC_particles", "particle_vx"].v
        pvx = (pvx_raw * (cm / s)).to("km/s").v

        fig_ph, ax_ph = plt.subplots(figsize=(9, 5))
        sc = ax_ph.scatter(px, pvx, s=0.4, c=np.abs(pvx),
                           cmap="plasma",
                           norm=Normalize(0, np.percentile(np.abs(pvx), 99)),
                           alpha=0.7, rasterized=True)
        ax_ph.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax_ph.set_title(f"Phase space  |  $a = {a_now:.3f}$", fontweight="bold")
        ax_ph.set_xlabel("x [kpc]"); ax_ph.set_ylabel(r"$v_x$ [km s$^{-1}$]")
        ax_ph.set_xlim(0, Lx)
        ax_ph.grid(True, linestyle=":", alpha=0.5)
        cbar = fig_ph.colorbar(sc, ax=ax_ph,
                               label=r"$|v_x|$ [km s$^{-1}$]", shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
        fig_ph.tight_layout()
        fig_ph.savefig(os.path.join(save_path, "PhaseSpace",
                                    f"phase_{i:04d}.png"), dpi=150)
        plt.close(fig_ph)

        frames_phase.append((px.copy(), pvx.copy(), a_now))

        # ---- 3D particle scatter ----
        py = ad["CIC_particles", "particle_position_y"].to("kpc").v
        pz = ad["CIC_particles", "particle_position_z"].to("kpc").v

        n_sub = min(20000, len(px))
        idx3d = np.random.choice(len(px), n_sub, replace=False)

        fig3 = plt.figure(figsize=(7, 6))
        ax3  = fig3.add_subplot(111, projection="3d")
        sc3  = ax3.scatter(px[idx3d], py[idx3d], pz[idx3d],
                           s=0.3, c=np.abs(pvx[idx3d]),
                           cmap="plasma", alpha=0.7, rasterized=True)
        fig3.colorbar(sc3, ax=ax3, label=r"$|v_x|$ [km/s]", shrink=0.6)
        ax3.set_xlabel("x [kpc]"); ax3.set_ylabel("y [kpc]"); ax3.set_zlabel("z [kpc]")
        ax3.set_xlim(0, Lx); ax3.set_ylim(0, Ly); ax3.set_zlim(0, Lz)
        ax3.set_title(f"DM particles  |  $a = {a_now:.3f}$", fontweight="bold")
        ax3.view_init(elev=22, azim=30 + 60 * i / max(len(plotfiles) - 1, 1))
        fig3.tight_layout()
        fig3.savefig(os.path.join(save_path, "ParticlePositions",
                                  f"3D_{i:04d}.png"), dpi=120)
        frames_3d.append((px[idx3d].copy(), py[idx3d].copy(), pz[idx3d].copy(),
                          pvx[idx3d].copy(), a_now, float(ds.current_time)))
        plt.close(fig3)

        print(f"[{i+1:3d}/{len(plotfiles)}] a={a_now:.4f}  z={z_now:.2f}"
              f"  delta_max={delta_max:.3f}  L2_err={l2:.4f}"
              f"  |vx|_max={np.max(np.abs(pvx)):.1f} km/s")

    # ---- Finalize history plot ----
    ax_hist.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5, label=r"$\delta=0$")
    ax_hist.set_title("DM density contrast evolution — Zel'dovich pancake", fontweight="bold", pad=12)
    ax_hist.set_xlabel("x [kpc]"); ax_hist.set_ylabel(r"$\delta$")
    ax_hist.set_xlim(0, Lx)
    ax_hist.grid(True, ls=":", alpha=0.5)
    handles, labels = ax_hist.get_legend_handles_labels()
    step = max(1, len(handles) // 8)
    ax_hist.legend(handles[::step], labels[::step], fontsize=8, loc="upper right", framealpha=0.85, ncol=2)
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join(save_path, "DeltaHistory", "delta_history.png"), dpi=300)
    plt.close(fig_hist)
    print(f"\nDelta history saved.")

    # ---- delta_max(a): numerical peak vs analytical peak ----
    # The analytical peak is evaluated at q_center=0 (where cos=1):
    #   delta_max(a) = 1/(1 - A) - 1 = A/(1-A) = (a/a_c)/(1 - a/a_c)
    a_arr = np.array(a_values)
    dmax  = np.array(delta_max_vals)
    a_lin = np.linspace(a_arr[0], min(a_arr[-1], 0.9 * a_collapse), 200)  # analyical solution a sampling (before a_collapse)
    a_lin_early = np.linspace(a_arr[0], 0.7 * a_collapse, 100)            # early linear solution a sampling

    fig_g, ax_g = plt.subplots(figsize=(8, 5))
    ax_g.plot(a_arr, dmax, "o-", ms=4, lw=1.5, label=r"$\delta_\mathrm{max}$ numerical")   # max delta evolution
    ax_g.plot(a_lin, a_lin / a_collapse / (1 - a_lin / a_collapse), ls="--", lw=1.5, color="crimson", label=r"Analytical $\delta_\mathrm{max}$")
    ax_g.plot(a_lin_early, a_lin_early / a_collapse, ls=":", color="seagreen", lw=1.5, label=r"Linear growth $\delta \propto a/a_c$")
    ax_g.axvline(a_collapse, color="gray", ls=":", lw=1, label=r"$a_\mathrm{collapse}$")  # vertical line at a_collapse
    ax_g.set_xlabel("Scale factor $a$")
    ax_g.set_ylabel(r"$\delta_\mathrm{max}$")
    ax_g.set_title("Peak density contrast growth", fontweight="bold")
    ax_g.legend()
    ax_g.grid(True, ls=":", alpha=0.5)
    fig_g.tight_layout()

    fig_g.savefig(os.path.join(save_path, "DeltaHistory", "delta_max_growth.png"), dpi=300)
    plt.close(fig_g)

    # ---- L2 profile error vs a ----
    l2_arr = np.array(l2_errors)

    fig_l2, ax_l2 = plt.subplots(figsize=(8, 5))
    ax_l2.semilogy(a_arr, l2_arr, "s-", ms=5, lw=1.5, color="#2ca02c", label=r"$\|\delta_\mathrm{num} - \delta_\mathrm{anal}\|_2$")
    ax_l2.axvline(a_collapse, color="gray", ls=":", lw=1, label=r"$a_\mathrm{collapse}$")
    ax_l2.set_xlabel("Scale factor $a$")
    ax_l2.set_ylabel(r"L2 profile error  $\sqrt{\langle(\delta_\mathrm{num}-\delta_\mathrm{anal})^2\rangle}$")
    ax_l2.set_title(r"Density profile L2 error", fontweight="bold")
    ax_l2.legend()
    ax_l2.grid(True, ls=":", alpha=0.5, which="both")
    fig_l2.tight_layout()

    fig_l2.savefig(os.path.join(save_path, "DeltaHistory", "delta_profile_l2_error.png"), dpi=300)
    plt.close(fig_l2)
    print("L2 profile error plot saved.")

    # ---- Animation: delta profile ----
    if len(frames_delta) > 1:
        fig_an, ax_an = plt.subplots(figsize=(10, 5))
        line_num,  = ax_an.plot([], [], lw=2, color="#1f77b4", label="Quokka")
        line_anal, = ax_an.plot([], [], lw=2, ls="--", color="crimson",
                                label="Analytical")
        ax_an.set_xlim(0, Lx); ax_an.set_ylim(-1.5, 20)
        ax_an.set_xlabel("x [kpc]"); ax_an.set_ylabel(r"$\delta$")
        ax_an.legend(); ax_an.grid(True, ls=":", alpha=0.5)
        title_an = ax_an.set_title("")

        def update_delta(frame):
            x_num, delta_num, x_an, delta_an, a, time = frames_delta[frame]
            line_num.set_data(x_num, delta_num)
            line_anal.set_data(x_an, delta_an)
            ax_an.set_ylim(-1.5, min(
                max(np.max(delta_num), np.max(delta_an)) * 1.3 + 0.5, 30))
            td_myr = time / (365.25 * 24.0 * 3600.0 * 1e6)
            title_an.set_text(
                f"DM density contrast  |  $a = {a:.3f}$, $t = {td_myr:.1f}$ Myr")
            return line_num, line_anal, title_an

        ani = animation.FuncAnimation(fig_an, update_delta,
                                      frames=len(frames_delta),
                                      interval=200, blit=False)
        ani_path = os.path.join(save_path, "Animations", "delta_evolution.gif")
        ani.save(ani_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_an)
        print(f"Delta animation saved: {ani_path}")

    # ---- Animation: 2D density map ----
    if len(frames_2d) > 1:
        init_map, init_xe, init_ye, init_a, _ = frames_2d[0]
        vmin_ani = max(init_map[init_map > 0].min(), 1e-3)
        vmax_ani = max(f[0].max() for f in frames_2d)

        fig_2da, ax_2da = plt.subplots(figsize=(8, 7))
        im_2da = ax_2da.imshow(
            init_map.T, origin="lower",
            extent=[init_xe[0], init_xe[-1], init_ye[0], init_ye[-1]],
            cmap="viridis",
            norm=(LogNorm(vmin=vmin_ani, vmax=vmax_ani)
                  if vmax_ani / vmin_ani > 10
                  else Normalize(vmin=vmin_ani, vmax=vmax_ani)),
            animated=True)
        ax_2da.set_xlabel("X [kpc]", fontsize=11)
        ax_2da.set_ylabel("Y [kpc]", fontsize=11)
        cbar = fig_2da.colorbar(im_2da, ax=ax_2da, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Sigma_\mathrm{DM}$ [particles kpc$^{-2}$]", fontsize=10)
        title_2da = ax_2da.set_title("", fontsize=12, fontweight="bold")

        def update_2d_map(frame):
            d_map, xe, ye, a_val, t_val = frames_2d[frame]
            im_2da.set_array(d_map.T)
            t_myr = t_val / (365.25 * 24.0 * 3600.0 * 1e6)
            title_2da.set_text(
                f"DM Surface Density  |  $a = {a_val:.3f}$, $t = {t_myr:.1f}$ Myr")
            return [im_2da, title_2da]

        ani_2d = animation.FuncAnimation(fig_2da, update_2d_map,
                                         frames=len(frames_2d),
                                         interval=200, blit=False)
        ani_2d_path = os.path.join(save_path, "Animations", "2D_DM_collapse.gif")
        ani_2d.save(ani_2d_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_2da)
        print(f"2D Density animation saved: {ani_2d_path}")

    # ---- Animation: phase space ----
    if len(frames_phase) > 1:
        fig_pa, ax_pa = plt.subplots(figsize=(9, 5))
        sc_pa = ax_pa.scatter([], [], s=0.5, c=[], cmap="plasma",
                              vmin=0, vmax=300, alpha=0.7, rasterized=True)
        ax_pa.set_xlim(0, Lx)
        ax_pa.set_xlabel("x [kpc]"); ax_pa.set_ylabel(r"$v_x$ [km s$^{-1}$]")
        title_pa = ax_pa.set_title("")
        ax_pa.grid(True, ls=":", alpha=0.4)
        cbar_pa = fig_pa.colorbar(sc_pa, ax=ax_pa,
                                  label=r"$|v_x|$ [km s$^{-1}$]", shrink=0.8)
        cbar_pa.ax.tick_params(labelsize=9)
        fig_pa.tight_layout()

        def update_phase(frame):
            xp, vp, av = frames_phase[frame]
            step = max(1, len(xp) // 5000)
            sc_pa.set_offsets(np.c_[xp[::step], vp[::step]])
            sc_pa.set_array(np.abs(vp[::step]))
            ax_pa.set_ylim(np.min(vp) * 1.1, np.max(vp) * 1.1)
            title_pa.set_text(f"Phase space  |  $a = {av:.3f}$")
            return sc_pa, title_pa

        ani_p = animation.FuncAnimation(fig_pa, update_phase,
                                        frames=len(frames_phase),
                                        interval=200, blit=False)
        ani_p_path = os.path.join(save_path, "Animations", "phase_evolution.gif")
        ani_p.save(ani_p_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_pa)
        print(f"Phase animation saved: {ani_p_path}")

    # ---- Animation: 3D DM particles with rotation ----
    if len(frames_3d) > 1:
        fig_3da = plt.figure(figsize=(8, 7))
        ax_3da  = fig_3da.add_subplot(111, projection="3d")
        sc_3da  = ax_3da.scatter([], [], [], s=0.3, c=[], cmap="plasma",
                                 alpha=0.7, rasterized=True)
        ax_3da.set_xlabel("x [kpc]"); ax_3da.set_ylabel("y [kpc]")
        ax_3da.set_zlabel("z [kpc]")
        ax_3da.set_xlim(0, Lx); ax_3da.set_ylim(0, Ly); ax_3da.set_zlim(0, Lz)
        all_v_max = max(np.max(np.abs(f[3])) for f in frames_3d)
        sc_3da.set_clim(0, all_v_max)
        fig_3da.colorbar(sc_3da, ax=ax_3da, label=r"$|v_x|$ [km/s]", shrink=0.6)
        title_3da = ax_3da.set_title("")

        def update_3d(frame):
            x3, y3, z3, v3, av, t3 = frames_3d[frame]
            sc_3da._offsets3d = (x3, y3, z3)
            sc_3da.set_array(np.abs(v3))
            ax_3da.view_init(elev=22,
                             azim=30 + 60 * frame / max(len(frames_3d) - 1, 1))
            t_myr = t3 / (365.25 * 24 * 3600 * 1e6)
            title_3da.set_text(
                f"DM particles  |  $a = {av:.3f}$, $t = {t_myr:.1f}$ Myr")
            return sc_3da, title_3da

        ani_3d = animation.FuncAnimation(fig_3da, update_3d,
                                         frames=len(frames_3d),
                                         interval=200, blit=False)
        ani_3d_path = os.path.join(save_path, "Animations",
                                   "particles_3d_evolution.gif")
        ani_3d.save(ani_3d_path, writer="pillow", fps=3, dpi=120)
        plt.close(fig_3da)
        print(f"3D Particles animation saved: {ani_3d_path}")

    # ---- Summary panel ----
    frames_delta_a = [frame[4] for frame in frames_delta]
    idx_collapse   = np.argmin(np.abs(np.array(frames_delta_a) - a_collapse))  # idx of collapse

    fig_sum, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig_sum.suptitle("DMZeldovich — Summary Panel", fontweight="bold", fontsize=14)

    # Panel 1: delta(x) history
    for idx, (xc, dc) in enumerate(zip(x_coords_hist, delta_hist)):
        axs[0, 0].plot(xc, dc, alpha=0.7, lw=0.9,
                       color=cmap(idx / max(len(delta_hist) - 1, 1)))
    axs[0, 0].axhline(0, color="k", ls="--", lw=0.7, alpha=0.4)
    axs[0, 0].set_title(r"$\delta(x)$ evolution (all snapshots)")
    axs[0, 0].set_xlabel("x [kpc]"); axs[0, 0].set_ylabel(r"$\delta$")
    axs[0, 0].set_xlim(0, Lx)

    # Panel2: delta_max(a) with linear and analytical regime
    axs[0, 1].plot(a_arr, dmax, "o-", ms=4, lw=1.5,
                   label=r"$\delta_\mathrm{max}$ numerical")
    a_lin2 = np.linspace(a_arr[0], min(a_arr[-1], 0.9 * a_collapse), 200)
    axs[0, 1].plot(a_lin2, a_lin2 / a_collapse / (1 - a_lin2 / a_collapse),
                   ls="--", color="crimson", label=r"Analytical $\max\delta(x)$")
    a_lin_e = np.linspace(a_arr[0], 0.3 * a_collapse, 100)
    axs[0, 1].plot(a_lin_e, a_lin_e / a_collapse,
                   ls=":", color="seagreen", lw=1.5,
                   label=r"Linear $\delta \propto a$")
    axs[0, 1].axvline(a_collapse, color="gray", ls=":", lw=1)
    axs[0, 1].set_xlabel("$a$"); axs[0, 1].set_ylabel(r"$\delta_\mathrm{max}$")
    axs[0, 1].set_title("Peak density contrast growth")
    axs[0, 1].legend(fontsize=8)

    # Panel 3: phase space at collapse
    xf, vf, _ = frames_phase[idx_collapse]
    step_f = max(1, len(xf) // 10000)
    sc_s = axs[1, 0].scatter(xf[::step_f], vf[::step_f], s=0.3,
                              c=np.abs(vf[::step_f]), cmap="plasma",
                              norm=Normalize(0, np.percentile(np.abs(vf), 99)),
                              alpha=0.6, rasterized=True)
    fig_sum.colorbar(sc_s, ax=axs[1, 0],
                     label=r"$|v_x|$ [km/s]", shrink=0.8)
    axs[1, 0].set_xlabel("x [kpc]"); axs[1, 0].set_ylabel(r"$v_x$ [km/s]")
    axs[1, 0].set_title(f"Phase space at collapse ($a = {a_collapse:.1f}$)")
    axs[1, 0].set_xlim(0, Lx)

    # Panel 4: delta(x) comparison 5 snap before collapse
    xn, dn, xa, da, a_before_coll, _ = frames_delta[idx_collapse - 5]
    axs[1, 1].plot(xn, dn, lw=2, color="#1f77b4", label="Quokka")
    axs[1, 1].plot(xa, da, lw=2, ls="--", color="crimson", label="Analytical")
    axs[1, 1].set_xlabel("x [kpc]"); axs[1, 1].set_ylabel(r"$\delta$")
    axs[1, 1].set_title(rf"Before collapse: \delta(x) comparison  ($a = {a_before_coll:.1f}$)")
    axs[1, 1].set_xlim(0, Lx)
    axs[1, 1].legend(fontsize=9)

    for ax in axs.flat:
        ax.grid(True, ls=":", alpha=0.4)
    fig_sum.tight_layout()
    fig_sum.savefig(os.path.join(save_path, "summary_panel.png"), dpi=250)
    plt.close(fig_sum)
    print("Summary panel saved.")

    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()

