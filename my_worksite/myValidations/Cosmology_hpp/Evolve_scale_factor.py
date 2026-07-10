# Validate the integrator of the scale factor in Cosmology.hpp

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
matplotlib.use('Agg')   # force matplotlib not to use the graphics
from numba import njit, prange


# 1. Hubble function: E(a) = sqrt(Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_L)
@njit
def HubbleFunction(a, Omega_r, Omega_m, Omega_lambda):
    """ Compute the Hubble function according to the energy density budget
        of the Universe"""

    # 1a. curvature parameter for difference
    Omega_K = 1 - Omega_r - Omega_m - Omega_lambda 

    val = Omega_r * a**-4 + Omega_m * a**-3 + Omega_K * a**-2 + Omega_lambda
    return np.sqrt(np.maximum(0.0, val))   # as in Quokka, np.maximum guards against sqrt of a tiny negative number from floating-point round-off



# 2. Evolve the scale factor a(t) from a_old to a_old+dt using an RK2 sub-stepped integrator
@njit
def evolveScaleFactor(a_init, dt, H0, Omega_r, Omega_m, Omega_l, max_frac_step):
    """ Reply the calculation done in Quokka and evolve the scale factor 
        according to the energy density budget. Method: RK2 sub-stepped integrator.
        Return a and nsteps, the numeber of steps required for a (prop to H_est) 
        does't vary more than max_frac_step"""
    
    # 2a. Estimate how many sub-steps: H*dt < max_frac_step per sub-step
    H_est = H0 * HubbleFunction(a_init, Omega_r, Omega_m, Omega_l)

    # Number of a steps (nsteps) required for a (prop to H_est) does't vary more than max_frac_step
    nsteps = max(1, int(np.ceil(H_est * dt / max_frac_step)))      # np.ceil rounds up to the clostest integer
    dt_sub = dt / nsteps

    # 2b. RK2 midpoint integrator 
    a = a_init  # strarting scale factor value for the integrator

    for _ in range(nsteps):
        k1 = a * H0 * HubbleFunction(a, Omega_r, Omega_m, Omega_l)          # da/dt at t
        a_mid = a + 0.5 * dt_sub * k1                                       # a at t+dt/2
        k2 = a_mid * H0 * HubbleFunction(a_mid, Omega_r, Omega_m, Omega_l)  # da/dt at t+dt/2
        a += dt_sub * k2
    return a, nsteps
       
    
# 3. Check against scipy.integrate

# 3a. Lhs Friedmann equation
def dadt(t, a, H0, Om, Or, OL):
    """Define the lhs of the first Friedmann equation"""
    return a * H0 * HubbleFunction(a, Or, Om, OL)

# 3b. Scipy solver:  explicit RK45’(default)
def get_sol_sci(lhs, dt, a_init, H0, Om, Or, OL, rtol=1e-12):
    sci_sol = solve_ivp(lhs, [0, dt], [a_init], args=(H0, Om, Or, OL), rtol=rtol)
    a_exact = sci_sol.y[0, -1]   # first row (here only 1, the only variable), last value, the final a
    return sci_sol, a_exact


# ------ Validation of  the cosmological expansion test ------

# 1. Physical parameters of the test
Or = 0
Om = 1            # EdS universe
OL = 0
h = 0.7
a_start = 1e-3

# From the input file of the test
yr_to_s = 3.15576e7
dt = 2.945e11 * yr_to_s    # from a=0.001 to a=10
cosmology_dt_limit = 1e-4

# From the cosmology ParmParse of QuokkaSimulation.hpp
Mpc_to_cm = 3.08567758e24
H0 = (h * 100.0 * 1e5) / Mpc_to_cm   # cgs


# 2. Quokka-like calculation
# 2a. Mimic the number of quokka plotfiles for the time values
n_points = 744                       # from plotfile_interval = 500 in the inputs 
times = np.linspace(0, dt, n_points) # mimic a equispaced (no cfl refinement included) quokka time array 

# 2b. Creation of a array of the dimension of the number of plotfiles of the simulation
a_quokka = np.zeros_like(times)
a_quokka[0] = a_start

# 2c. Integration with RK between each fictious plotfile
for i in range(1, len(times)):
    dt_intervallo = times[i] - times[i-1] # separation between contiguos plotfiles
    a_quokka[i], _ = evolveScaleFactor(a_quokka[i-1], dt_intervallo, H0, Or, Om, OL, cosmology_dt_limit)

# a_quokka, nsteps_quokka = evolveScaleFactor(a_start, dt, H0, Or, Om, OL, cosmology_dt_limit)   # max_frac_timestep = 0.01


# 3. Consistency analysis and fit
# 3a. Theoretical values
t_start_eds = (2 / (3 * H0)) * (a_start**1.5) 
a_theoretical = (1.5 * H0 * (t_start_eds + times))**(2/3)  # theoretical values

# 3b. Relative error
error_rel = np.abs(a_quokka - a_theoretical) / a_theoretical 
print(f"Max consistency error in validation: {np.max(error_rel):.2e}")

# 3c. Linear fit (Log-Log)
log_a = np.log10(a_quokka)
log_t = np.log10(t_start_eds + times)
slope, intercept = np.polyfit(log_t, log_a, 1)   # 1: linear log(a) = intercept + slope * log(t)

print(f"Power law analysis (Validation script):")
print(f"Theoretical: 0.6667 | Fit: {slope:.4f}")
print(f"Difference: {np.abs(2/3 - slope):.2e}")


# 4. Visualization
fig, ax = plt.subplots()
ax.plot(times, a_quokka, label="a(t)")

ax.set_title("Scale factor evolution: a(t) (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("t [s]", fontsize=11)
ax.set_ylabel("Scale factor (a)", fontsize=11)
ax.tick_params(direction='in', which='both')   # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', bbox_to_anchor=(0.25, 0.85))

# Insertion
ax_ins = inset_axes(ax, width='30%', height = '25%', loc='lower right', borderpad=3)  # 30% wide and 25% high of the main chart
ax_ins.plot(a_quokka, np.abs(error_rel), color='red', lw=1) # np.abs for log scale
ax_ins.set_title(r"Scale factor RE: $\frac{a_{\mathrm{sim}} - a_{\mathrm{pre}}}{a_{\mathrm{pre}}}$", fontsize=9)
ax_ins.tick_params(axis='both', labelsize=8)
ax_ins.grid(True, linestyle=':', alpha=0.5)
ax_ins.tick_params(direction='in', which='both')          # ticks point inward
ax_ins.axhline(0, color='black', lw=0.5, linestyle='--')  # reference line

# Text insertion in the figure
stats_text = (f'Fit slope: {slope:.4f} (expected: {2 / 3:.2f})')
ax.text(0.05, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myValidations/Cosmology_hpp/validation_outputs/Validation_ScaleFactorIntegrator.png', dpi=300)
print(f"Scale Factor integration results saved to: /data/mfulghieri/quokka/my_worksite/myValidations/Cosmology_hpp/validation_outputs")



# 5. Scipy calculation
_, a_RK45 = get_sol_sci(dadt, dt, a_start, H0, Om, Or, OL)

# Scipy comparison prints
print(f"Quokka-like: {a_quokka[-1]:.10f}")
print(f"Scipy:  {a_RK45:.10f}")
print(f"Relative error: {abs(a_quokka[-1] - a_RK45)/a_RK45:.2e}")



# ------ Study of the precision varying max_frac_timestep ------
# In the test, this corresponds to static constexpr double cosmology_dt_limit

# 1. Simulation of a(t) for different max_frac_timestep
@njit(parallel=True) # parallel=True enables the computations across multiple cores
def study_precision(limits, a_start, times, H0, Or, Om, OL):  # limits: array of the max_frac_timestep to be tested
    """Simulation of a(t) for different max_frac_timestep"""
    n_limits = len(limits)
    n_times = len(times)
    results = np.zeros((n_limits, n_times))  # matrix of the a(t) solution for each max_frac_timestep
    
    # Loop on every max_frac_timesteps (mft)
    for j in prange(n_limits):   # parallel for cycle
        limit = limits[j]        # test the j mft for this simulation
        a_current = a_start
        results[j, 0] = a_start  # initiali a(t) condition for the j-th mft
        # Integration of the scale factor over each timestep
        for i in range(1, n_times):
            dt_step = times[i] - times[i-1]
            # Call the RK2 solver
            a_next, _ = evolveScaleFactor(a_current, dt_step, H0, Or, Om, OL, limit)
            results[j, i] = a_next
            a_current = a_next
    return results



# 2. Parameters of the mft
mft = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])


# 3. Simulation of a(t) for the mft
a_results = study_precision(mft, a_start, times, H0, Or, Om, OL)


# 4. Visualization
fig_conv, ax_conv = plt.subplots(figsize=(8, 6))

# 4.1 Relative error
for i in range(len(mft)):
    err = np.abs(a_results[i, :] - a_theoretical) / a_theoretical
    # Plot each relative error
    ax_conv.plot(a_theoretical, err, label=f'limit = {mft[i]:.0e}')

# 5. Formattation
ax_conv.set_title("Convergence Study: RK2 Precision vs max_frac_timestep", 
                  pad=25, fontsize=14, fontweight='bold')
ax_conv.set_xlabel("Scale Factor (a)", fontsize=11)
ax_conv.set_ylabel("Relative Error", fontsize=11)
ax_conv.set_xscale('log')
ax_conv.set_yscale('log')
ax_conv.tick_params(direction='in', which='both', top=True, right=True)
ax_conv.grid(True, which='both', linestyle=':', alpha=0.5)
ax_conv.legend(loc='best', frameon=True, fontsize=10)


fig_conv.tight_layout()
output_path = '/data/mfulghieri/quokka/my_worksite/myValidations/Cosmology_hpp/validation_outputs/Convergence_Study_RK2.png'
fig_conv.savefig(output_path, dpi=300)
print(f"Covergency study saved in: {output_path}")



# RK4 manuale sia python che Cosmology.hpp



