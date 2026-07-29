#!/usr/bin/env python3
"""
analyze_dmZeldovich.py
======================
Scientific analysis script for the DM Zel'dovich pancake test (Quokka).

Produces:
  - History of DM density contrast evolution (all snapshots on one figure)
  - Per-snapshot comparison: numerical vs analytical Zel'dovich profile
  - Per-snapshot phase-space diagram (vx vs x)
  - Per-snapshot 2D gas density slice with velocity quiver
  - 3D particle scatter (one per snapshot, azimuth-rotating animation)
  - MP4 animation of density profile evolution
  - MP4 animation of phase-space evolution

Usage:
    python analyze_dmZeldovich.py --plotfiles /path/to/output/plt* --save /path/to/save
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
from unyt import cm, km, s, Mpc
from datetime import datetime

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
               "ParticlePositions", "Hydro", "Animations"]
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

    # Consistency with the test    
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

    plotfiles = [p for p in plotfiles if not p.endswith(".old")]

    if not plotfiles:
        print("No plotfiles found!"); sys.exit(1)
    print(f"Found {len(plotfiles)} plotfiles.")

    a_collapse = args.a_collapse

    # --- Storage for history ---
    a_values        = []
    times           = []
    delta_max_vals  = []
    x_coords_hist   = []
    delta_hist      = []

    # Figures that accumulate across snapshots
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("coolwarm_r")

    # Storage for animations 
    frames_delta  = []   # (x_coord, delta, x_anal, delta_anal, a_val. current_time)
    frames_2d     = []   # (density_map, a_now, current_time)
    frames_phase  = []   # (px, pvx, a_val)
    frames_hydro  = []   # list of PNG yt files
    frames_3d     = []   # (px, py, pz, vx, a_now, current_time)

    for i, plt_path in enumerate(plotfiles):
        ds  = yt.load(plt_path)
        ad  = ds.all_data()
        meta = load_metadata(plt_path)
        cosmo = get_cosmo(meta)

        a_now  = cosmo.get("a", float(ds.current_time))   # fallback to time
        z_now  = cosmo.get("z", (1.0 / a_now - 1.0) if a_now > 0 else 0.0)
        H0_cgs = cosmo.get("H0", 2.27e-18)                # 70 km/s/Mpc in cgs

        a_values.append(a_now)
        times.append(float(ds.current_time))

        # Physical box size
        Lx = float(ds.domain_width[0].to("Mpc").v)

        # ---- 1D DM density profile along x ----
        c_vals  = ds.domain_center.v         # [x_c, y_c, z_c]
        left_v  = ds.domain_left_edge.v      # [0, 0, 0]
        right_v = ds.domain_right_edge.v     # [Lx, Ly, Lz]
        ray = ds.ray([left_v[0], c_vals[1], c_vals[2]],    # from left to right in x and center in y and z
                     [right_v[0], c_vals[1], c_vals[2]])
        sort_idx  = np.argsort(ray["index", "x"])
        x_coord   = ray["index", "x"][sort_idx].to("Mpc").v
        rho_dm    = ray["deposit", "CIC_particles_density"][sort_idx].v
        rho_dm_mean = np.mean(rho_dm) if np.mean(rho_dm) > 0 else 1.0
        delta     = rho_dm / rho_dm_mean - 1.0
        delta_max = float(np.max(delta))

        delta_max_vals.append(delta_max)
        x_coords_hist.append(x_coord)
        delta_hist.append(delta)

        color_val = cmap(i / max(len(plotfiles) - 1, 1))

        # History plot (accumulate)
        ax_hist.plot(x_coord, delta, color=color_val,
                     lw=1.2, label=f"a={a_now:.3f}")

        # ---- Analytical solution ----
        x_anal, delta_anal, v_anal = zeldovich_analytical(
            a_now, a_collapse, Lx, H0_cgs)

        # ---- Per-snapshot comparison: numerical vs analytical ----
        fig_c, ax_c = plt.subplots(figsize=(10, 5))
        ax_c.plot(x_coord, delta, lw=2, color="#1f77b4",
                  label="Quokka (numerical)")
        ax_c.plot(x_anal, delta_anal, lw=2, ls="--", color="crimson",
                  label="Zel'dovich (analytical)")
        ax_c.set_title(f"DM density contrast  |  $a = {a_now:.3f}$,  $z = {z_now:.2f}$",
                       fontweight="bold")
        ax_c.set_xlabel("x [Mpc]"); ax_c.set_ylabel(r"$\delta = \rho/\bar\rho - 1$")
        ax_c.set_xlim(0, Lx)
        ax_c.set_ylim(max(-1.5, min(delta) * 1.3),
                      min(delta_max * 2.0 + 0.5, 50))
        ax_c.legend(); ax_c.grid(True, ls=":", alpha=0.5)
        fig_c.tight_layout()
        fig_c.savefig(os.path.join(save_path, "DeltaComparison",
                                   f"comparison_{i:04d}.png"), dpi=200)
        plt.close(fig_c)

        # Store for animation
        frames_delta.append((x_coord.copy(), delta.copy(),   # .copy() to avoid pointer and so overwrite data at nest step
                              x_anal.copy(), delta_anal.copy(), a_now, float(ds.current_time)))

        # --- 2D Histogram of Particle Density (X-Y Plane) ---
        px      = ad["CIC_particles", "particle_position_x"].to("Mpc").v
        py      = ad["CIC_particles", "particle_position_y"].to("Mpc").v
        rho_box = ad["deposit", "CIC_particles_density"].v
        bins = 512

        density_map, xedges, yedges = np.histogram2d(px, py, 
                    bins=bins,
                    #weights=rho_box,
                    range=[[0, Lx], [0, float(ds.domain_width[1].to("Mpc").v)]])
        
        fig2d, ax2d =  plt.subplots(figsize=(6, 5))
        im = ax2d.imshow(density_map.T, origin="lower", 
                         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         cmap="viridis",
                         norm=Normalize(vmin=density_map[density_map > 0].min(), vmax=density_map.max()))
        
        ax2d.set_title(f"DM Density Map ($a = {a_now:.3f}$)", fontsize=12, fontweight='bold')
        ax2d.set_xlabel("X [Mpc]", fontsize=11)
        ax2d.set_ylabel("Y [Mpc]", fontsize=11)
        cbar = fig2d.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
        cbar.set_label("Particle Count per Cell", fontsize=10)
        fig2d.savefig(os.path.join(save_path, "2D_DM_Density", f"Density2D_{i:03d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2d)

        # Store for animation
        frames_2d.append((density_map.copy(), a_now, float(ds.current_time)))

        # ---- Phase space ----
        px  = ad["CIC_particles", "particle_position_x"].to("Mpc").v
        pvx_raw = ad["CIC_particles", "particle_vx"].v
        pvx = (pvx_raw * (cm / s)).to("km/s").v

        fig_ph, ax_ph = plt.subplots(figsize=(9, 5))
        sc = ax_ph.scatter(px, pvx, s=0.4, c=np.abs(pvx),   # save sc for the colorbar
                           cmap="plasma", norm=Normalize(0, np.percentile(np.abs(pvx), 99)),
                           alpha=0.7, rasterized=True)
        ax_ph.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax_ph.set_title(f"Phase space  |  $a = {a_now:.3f}$", fontweight="bold")
        ax_ph.set_xlabel("x [Mpc]"); ax_ph.set_ylabel(r"$v_x$ [km s$^{-1}$]")
        ax_ph.set_xlim(0, Lx)
        ax_ph.grid(True, linestyle=':', alpha=0.5)
        cbar = fig_ph.colorbar(sc, ax=ax_ph, label=r"$|v_x|$ [km s$^{-1}$]", shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
        fig_ph.tight_layout()
        fig_ph.savefig(os.path.join(save_path, "PhaseSpace",
                                    f"phase_{i:04d}.png"), dpi=150)
        plt.close(fig_ph)

        frames_phase.append((px.copy(), pvx.copy(), a_now))

        # ---- 2D gas density slice ----
        slc = yt.SlicePlot(ds, "z", ("boxlib", "gasDensity"))
        slc.annotate_quiver(("boxlib", "x-GasMomentum"),
                            ("boxlib", "y-GasMomentum"), factor=14)
        slc.annotate_timestamp(corner="upper_left", draw_inset_box=True)
        gas_img_path = os.path.join(save_path, "Hydro", f"gasDensity_{i:04d}.png")
        slc.save(gas_img_path)
        
        frames_hydro.append(gas_img_path)

        # ---- 3D particle scatter ----
        py = ad["CIC_particles", "particle_position_y"].to("Mpc").v
        pz = ad["CIC_particles", "particle_position_z"].to("Mpc").v
        Ly = float(ds.domain_width[1].to("Mpc").v)
        Lz = float(ds.domain_width[2].to("Mpc").v)

        #n_sub = max(1, len(px) // 50)
        n_sub = min(20000, len(px))     # 2% of the particle, but commented to avoid mess
        idx3d = np.random.choice(len(px), n_sub, replace=False)   # random set of indices

        fig3, ax3 = plt.figure(figsize=(7, 6)), None
        ax3 = fig3.add_subplot(111, projection="3d")
        sc3 = ax3.scatter(px[idx3d], py[idx3d], pz[idx3d],
                          s=0.3, c=np.abs(pvx[idx3d]),            # size = s and coloration according to x velocity
                          cmap="plasma", alpha=0.7, rasterized=True)
        fig3.colorbar(sc3, ax=ax3, label=r"$|v_x|$ [km/s]", shrink=0.6)
        ax3.set_xlabel("x [Mpc]"); ax3.set_ylabel("y [Mpc]"); ax3.set_zlabel("z [Mpc]")
        ax3.set_xlim(0, Lx); ax3.set_ylim(0, Ly); ax3.set_zlim(0, Lz)
        ax3.set_title(f"DM particles  |  $a = {a_now:.3f}$", fontweight="bold")
        ax3.view_init(elev=22, azim=30 + 60 * i / max(len(plotfiles) - 1, 1))
        fig3.tight_layout()
        fig3.savefig(os.path.join(save_path, "ParticlePositions",
                                  f"3D_{i:04d}.png"), dpi=120)
        frames_3d.append((px[idx3d].copy(), py[idx3d].copy(), pz[idx3d].copy(), pvx[idx3d].copy(), a_now, float(ds.current_time)))
        plt.close(fig3)

        print(f"[{i+1:3d}/{len(plotfiles)}] a={a_now:.4f}  z={z_now:.2f}"
              f"  delta_max={delta_max:.3f}  |vx|_max={np.max(np.abs(pvx)):.1f} km/s")

    # ---- Finalize history plot ----
    ax_hist.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5, label=r"$\delta=0$")
    ax_hist.set_title("DM density contrast evolution — Zel'dovich pancake",
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

    # ---- delta_max vs a ----
    a_arr = np.array(a_values)
    dmax  = np.array(delta_max_vals)
    fig_g, ax_g = plt.subplots(figsize=(8, 5))
    ax_g.plot(a_arr, dmax, "o-", ms=4, lw=1.5, label=r"$\delta_\mathrm{max}$ (numerical)")
    a_lin = np.linspace(a_arr[0], min(a_arr[-1], 0.9 * a_collapse), 200)
    ax_g.plot(a_lin, a_lin / a_collapse / (1 - a_lin / a_collapse),
              ls="--", color="crimson", label=r"Zel'dovich $\delta_\mathrm{max}$ (analytical)")
    ax_g.axvline(a_collapse, color="gray", ls=":", lw=1, label=r"$a_\mathrm{collapse}$")
    ax_g.set_xlabel("Scale factor $a$"); ax_g.set_ylabel(r"$\delta_\mathrm{max}$")
    ax_g.set_title("Peak density contrast growth", fontweight="bold")
    ax_g.legend(); ax_g.grid(True, ls=":", alpha=0.5)
    fig_g.tight_layout()
    fig_g.savefig(os.path.join(save_path, "DeltaHistory",
                               "delta_max_growth.png"), dpi=300)
    plt.close(fig_g)

    # ---- Animation: delta profile ----
    if len(frames_delta) > 1:
        fig_an, ax_an = plt.subplots(figsize=(10, 5))
        line_num, = ax_an.plot([], [], lw=2, color="#1f77b4", label="Quokka")
        line_anal, = ax_an.plot([], [], lw=2, ls="--", color="crimson",
                                label="Analytical")
        ax_an.set_xlim(0, Lx); ax_an.set_ylim(-1.5, 20)
        ax_an.set_xlabel("x [Mpc]"); ax_an.set_ylabel(r"$\delta$")
        ax_an.legend(); ax_an.grid(True, ls=":", alpha=0.5)
        title_an = ax_an.set_title("")

        def update_delta(frame):  #frames_delta.append((x_coord.copy(), delta.copy(),   # .copy() to avoid pointer and so overwrite data at nest step
                                   # x_anal.copy(), delta_anal.copy(), a_now, float(ds.current_time)))
            x_num, delta_num, x_an, delta_an, a, time = frames_delta[frame] 
            line_num.set_data(x_num, delta_num)
            line_anal.set_data(x_an, delta_an)
            ax_an.set_ylim(-1.5, min(max(np.max(delta_num), np.max(delta_an)) * 1.3 + 0.5, 30))
            td_myr =  time / (365.25 * 24.0 * 3600.0 * 1e6)
            title_an.set_text(f"DM density contrast  |  $a = {a:.3f}$, $t = {td_myr:.1f}$ Myr")
            return line_num, line_anal, title_an

        ani = animation.FuncAnimation(fig_an, update_delta,
                                      frames=len(frames_delta), interval=200, blit=False)
        ani_path = os.path.join(save_path, "Animations", "2D_DM_histo.gif")
        ani.save(ani_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_an)
        print(f"Delta animation saved: {ani_path}")

    # ---- Animation: 2D density histo ----
    if len(frames_2d) > 1:
        fig_2da, ax_2da = plt.subplots(figsize=(10, 5))

        init_map, init_a, init_t = frames_2d[0]
        im_2da = ax_2da.imshow(init_map.T, origin="lower",
                             extent=[0, Lx, 0, float(ds.domain_width[1].to("Mpc").v)],
                             cmap="viridis",
                             animated=True)
        ax_2da.set_xlabel("X [Mpc]", fontsize=11)
        ax_2da.set_ylabel("Y [Mpc]", fontsize=11)
        cbar = fig_2da.colorbar(im_2da, ax=ax_2da, fraction=0.046, pad=0.04)
        cbar.set_label("Particle Count per Cell", fontsize=10)
        title_2da = ax_2da.set_title("", fontsize=12, fontweight='bold')

        def update_2d_map(frame):
            d_map, a_val, t_val = frames_2d[frame]
            im_2da.set_array(d_map.T) 
            im_2da.set_clim(vmin=d_map[d_map > 0].min(), vmax=d_map.max())   # color dynamic autoscale
            t_myr = t_val / (365.25 * 24.0 * 3600.0 * 1e6) if t_val > 1e10 else t_val
            title_2da.set_text(f"DM Density Map  |  $a = {a_val:.3f}$, $t = {t_myr:.1f}$ Myr")
            return [im_2da, title_2da]
        
        ani = animation.FuncAnimation(fig_2da, update_2d_map,
                                      frames=len(frames_2d), interval=200, blit=False)
        
        ani_path = os.path.join(save_path, "Animations", "2D_DM_colcollapse.gif")
        ani.save(ani_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_2da)
        print(f"2D Density animation saved: {ani_path}")

    # ---- Animation: phase space ----
    if len(frames_phase) > 1:
        fig_pa, ax_pa = plt.subplots(figsize=(9, 5))
        sc_pa = ax_pa.scatter([], [], s=0.5, c=[], cmap="plasma",
                              vmin=0, vmax=300, alpha=0.7, rasterized=True)
        ax_pa.set_xlim(0, Lx)
        ax_pa.set_xlabel("x [Mpc]"); ax_pa.set_ylabel(r"$v_x$ [km s$^{-1}$]")
        title_pa = ax_pa.set_title("")
        ax_pa.grid(True, ls=":", alpha=0.4)
        # Colorbar
        cbar_pa = fig_pa.colorbar(sc_pa, ax=ax_pa, label=r"$|v_x|$ [km s$^{-1}$]", shrink=0.8)
        cbar_pa.ax.tick_params(labelsize=9)
        fig_pa.tight_layout()

        def update_phase(frame):
            xp, vp, av = frames_phase[frame]
            # Downsample for animation performance
            step = max(1, len(xp) // 5000)
            sc_pa.set_offsets(np.c_[xp[::step], vp[::step]])
            sc_pa.set_array(np.abs(vp[::step]))
            ax_pa.set_ylim(np.min(vp) * 1.1, np.max(vp) * 1.1)
            title_pa.set_text(f"Phase space  |  $a = {av:.3f}$")
            return sc_pa, title_pa

        ani_p = animation.FuncAnimation(fig_pa, update_phase,
                                        frames=len(frames_phase), interval=200, blit=False)
        ani_p_path = os.path.join(save_path, "Animations", "phase_evolution.gif")
        ani_p.save(ani_p_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_pa)
        print(f"Phase animation saved: {ani_p_path}")

    
     # ---- Animation: 3D DM particles with rotation ----
    if len(frames_3d) > 1:
        fig_3da = plt.figure(figsize=(8, 7))
        ax_3da = fig_3da.add_subplot(111, projection="3d")
        sc_3da = ax_3da.scatter([], [], [], s=0.3, c=[], cmap="plasma", alpha=0.7, rasterized=True) # initialize 3D empty plot
        ax_3da.set_xlabel("x [Mpc]"); ax_3da.set_ylabel("y [Mpc]"); ax_3da.set_zlabel("z [Mpc]")
        ax_3da.set_xlim(0, Lx); ax_3da.set_ylim(0, Ly); ax_3da.set_zlim(0, Lz)
        
        # Set colorbar limits (fixed)
        all_v_max = max(np.max(np.abs(f[3])) for f in frames_3d)
        sc_3da.set_clim(0, all_v_max)
        fig_3da.colorbar(sc_3da, ax=ax_3da, label=r"$|v_x|$ [km/s]", shrink=0.6)
        
        title_3da = ax_3da.set_title("")

        def update_3d(frame):
            x3, y3, z3, v3, av, t3 = frames_3d[frame]
            sc_3da._offsets3d = (x3, y3, z3) # update 3D positions
            sc_3da.set_array(np.abs(v3))     # update colors according to velocity
            current_azim = 30 + 60 * frame / max(len(frames_3d) - 1, 1)  # rotation angle of the frame
            ax_3da.view_init(elev=22, azim=current_azim)
            t_myr = t3 / (365.25 * 24 * 3600 * 1e6)
            title_3da.set_text(f"DM particles  |  $a = {av:.3f}$, $t = {t_myr:.1f}$ Myr")
            return sc_3da, title_3da

        ani_3d = animation.FuncAnimation(fig_3da, update_3d,
                                         frames=len(frames_3d), interval=200, blit=False)
        ani_3d_path = os.path.join(save_path, "Animations", "particles_3d_evolution.gif")
        ani_3d.save(ani_3d_path, writer="pillow", fps=3, dpi=120)
        plt.close(fig_3da)
        print(f"3D Particles animation saved: {ani_3d_path}")


    # ---- Animation: Gas Density (Hydro) ----
    if len(frames_hydro) > 1:
        from PIL import Image
        
        fig_ha, ax_ha = plt.subplots(figsize=(8, 8))
        ax_ha.set_xticks([])   # remeve ticks
        ax_ha.set_yticks([])
        
        # Load first image to initialize the object 
        first_png_path = frames_hydro[0]
        img_data = np.array(Image.open(first_png_path))      # take the first image
        img_obj = ax_ha.imshow(img_data)                     # convert the save paths 
        title_ha = ax_ha.set_title("")
        fig_ha.tight_layout()

        def update_hydro(frame):  # frame is integer increasing at every step
            # Read PNG of the current file and update the file
            img_path = frames_hydro[frame]            # extract path, a, t
            img_data = np.array(Image.open(img_path)) # read the data
            img_obj.set_data(img_data)                # substitute the pixels
            title_ha.set_text(f"Gas Density")
            return [img_obj, title_ha]

        ani_h = animation.FuncAnimation(fig_ha, update_hydro,
                                        frames=len(frames_hydro), interval=200, blit=True)
        ani_h_path = os.path.join(save_path, "Animations", "hydro_density_evolution.gif")
        ani_h.save(ani_h_path, writer="pillow", fps=3, dpi=150)
        plt.close(fig_ha)
        print(f"Hydro animation saved: {ani_h_path}")


    print(f"\nAll done. End at {datetime.now().time()}")


if __name__ == "__main__":
    main()



# python /data/mfulghieri/quokka/src/problems/DMZeldovich/analyze_dmZeldovich.py \
#        --plotfiles "/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_${SLURM_JOB_ID}/plt*" \
#        --save   "/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_${SLURM_JOB_ID}/analysis"
