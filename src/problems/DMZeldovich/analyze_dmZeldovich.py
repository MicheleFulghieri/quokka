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
    subdirs = ["DeltaHistory", "DeltaComparison", "PhaseSpace",
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
#   x(q) = q - (A(a)/k) * sin(k*q)      [collapse at x=L/2 for -sin]
#   rho(q) = rho_mean / |1 - A(a)*cos(k*q)|
#   v(q)   = -a * H(a) * (A(a)/k) * sin(k*q)
# ---------------------------------------------------------------------------
def zeldovich_analytical(a, a_collapse, Lx, H0, n_q=2000):
    """Return (x_analytical, delta_analytical, v_analytical) on a Lagrangian grid."""
    k  = 2.0 * np.pi / Lx
    A  = a / a_collapse
    H  = H0 * a**(-1.5)         # EdS: H(a) = H0 * a^{-3/2}

    q            = np.linspace(0.0, Lx, n_q, endpoint=False)
    x_analytical = q - (A / k) * np.sin(k * q)
    delta_analytical = 1.0 / (1.0 - A * np.cos(k * q)) - 1.0
    v_analytical     = -a * H * (A / k) * np.sin(k * q)
    return x_analytical, delta_analytical, v_analytical


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    save_path = args.save
    make_dirs(save_path)

    # --- Locate plotfiles ---
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
    frames_delta  = []   # (x_coord, delta, x_anal, delta_anal, a_val)
    frames_phase  = []   # (px, pvx, a_val)

    for i, plt_path in enumerate(plotfiles):
        ds  = yt.load(plt_path)
        ad  = ds.all_data()
        meta = load_metadata(plt_path)
        cosmo = get_cosmo(meta)

        a_now  = cosmo.get("a", float(ds.current_time))   # fallback to time
        z_now  = cosmo.get("z", (1.0 / a_now - 1.0) if a_now > 0 else 0.0)
        H0_cgs = cosmo.get("H0", 2.27e-18)                # ~70 km/s/Mpc in cgs

        a_values.append(a_now)
        times.append(float(ds.current_time))

        # Physical box size
        Lx = float(ds.domain_width[0].to("Mpc").v)

        # ---- 1D DM density profile along x ----
        c_vals  = ds.domain_center.v
        left_v  = ds.domain_left_edge.v
        right_v = ds.domain_right_edge.v
        ray = ds.ray([left_v[0], c_vals[1], c_vals[2]],
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
        frames_delta.append((x_coord.copy(), delta.copy(),
                              x_anal.copy(), delta_anal.copy(), a_now))

        # ---- Phase space ----
        px  = ad["CIC_particles", "particle_position_x"].to("Mpc").v
        pvx_raw = ad["CIC_particles", "particle_vx"].v
        pvx = (pvx_raw * (cm / s)).to("km/s").v

        fig_ph, ax_ph = plt.subplots(figsize=(9, 5))
        ax_ph.scatter(px, pvx, s=0.4, c=np.abs(pvx),
                      cmap="plasma", norm=Normalize(0, np.percentile(np.abs(pvx), 99)),
                      alpha=0.7, rasterized=True)
        ax_ph.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax_ph.set_title(f"Phase space  |  $a = {a_now:.3f}$", fontweight="bold")
        ax_ph.set_xlabel("x [Mpc]"); ax_ph.set_ylabel(r"$v_x$ [km s$^{-1}$]")
        ax_ph.set_xlim(0, Lx)
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
        slc.save(os.path.join(save_path, "Hydro", f"gasDensity_{i:04d}.png"))

        # ---- 3D particle scatter ----
        py = ad["CIC_particles", "particle_position_y"].to("Mpc").v
        pz = ad["CIC_particles", "particle_position_z"].to("Mpc").v
        Ly = float(ds.domain_width[1].to("Mpc").v)
        Lz = float(ds.domain_width[2].to("Mpc").v)

        n_sub = max(1, len(px) // 50)
        idx3d = np.random.choice(len(px), n_sub, replace=False)

        fig3, ax3 = plt.figure(figsize=(7, 6)), None
        ax3 = fig3.add_subplot(111, projection="3d")
        sc3 = ax3.scatter(px[idx3d], py[idx3d], pz[idx3d],
                          s=0.5, c=np.abs(pvx[idx3d]),
                          cmap="plasma", alpha=0.7, rasterized=True)
        fig3.colorbar(sc3, ax=ax3, label=r"$|v_x|$ [km/s]", shrink=0.6)
        ax3.set_xlabel("x [Mpc]"); ax3.set_ylabel("y [Mpc]"); ax3.set_zlabel("z [Mpc]")
        ax3.set_xlim(0, Lx); ax3.set_ylim(0, Ly); ax3.set_zlim(0, Lz)
        ax3.set_title(f"DM particles  |  $a = {a_now:.3f}$", fontweight="bold")
        ax3.view_init(elev=22, azim=30 + 60 * i / max(len(plotfiles) - 1, 1))
        fig3.tight_layout()
        fig3.savefig(os.path.join(save_path, "ParticlePositions",
                                  f"3D_{i:04d}.png"), dpi=120)
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

        def update_delta(frame):
            xc, dc, xa, da, av = frames_delta[frame]
            line_num.set_data(xc, dc)
            line_anal.set_data(xa, da)
            ax_an.set_ylim(-1.5, min(max(np.max(dc), np.max(da)) * 1.3 + 0.5, 30))
            title_an.set_text(f"DM density contrast  |  $a = {av:.3f}$")
            return line_num, line_anal, title_an

        ani = animation.FuncAnimation(fig_an, update_delta,
                                      frames=len(frames_delta), interval=200, blit=False)
        ani_path = os.path.join(save_path, "Animations", "delta_evolution.gif")
        ani.save(ani_path, writer="pillow", fps=5, dpi=150)
        plt.close(fig_an)
        print(f"Delta animation saved: {ani_path}")

    # ---- Animation: phase space ----
    if len(frames_phase) > 1:
        fig_pa, ax_pa = plt.subplots(figsize=(9, 5))
        sc_pa = ax_pa.scatter([], [], s=0.5, c=[], cmap="plasma",
                              vmin=0, vmax=300, alpha=0.7, rasterized=True)
        ax_pa.set_xlim(0, Lx)
        ax_pa.set_xlabel("x [Mpc]"); ax_pa.set_ylabel(r"$v_x$ [km s$^{-1}$]")
        title_pa = ax_pa.set_title("")
        ax_pa.grid(True, ls=":", alpha=0.4)

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
        ani_p.save(ani_p_path, writer="pillow", fps=5, dpi=150)
        plt.close(fig_pa)
        print(f"Phase animation saved: {ani_p_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()



# python /data/mfulghieri/quokka/src/problems/DMZeldovich/analyze_dmZeldovich.py \
#        --plotfiles "/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_${SLURM_JOB_ID}/plt*" \
#        --save   "/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_${SLURM_JOB_ID}/analysis"
