import yt
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Anti-Grain Geometry, backend without interface, allow to save the plots
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob
import yaml
from natsort import natsorted
from unyt import Mpc, km, s, g, cm



# --- Path to plotfiles ---
output_dir = "/data/mfulghieri/quokka/outputs/CosmologicalExpansion/CosmologicalExpansion_11965"

# retrive the plotfiles 
# os.path.join joins the folder path with the "plt*" pattern, glob.glob creates a list of all files/folders that begin with "plt"
plotfiles = glob.glob(os.path.join(output_dir, "plt*"))
#plotfiles = [f for f in glob.glob(os.path.join(output_dir, "plt[0-9]*")) if ".old" not in f]
plotfiles = natsorted(plotfiles)     # natural ordering to the list 

if not plotfiles: 
    print("No plotfiles found!")
    exit(1)


# --- Parameters ---

# 1. Dictionaries to stores the evolution
a_values  = []
times     = []
densities = []
energies  = []
momenta   = []
rho_spatial_std   = []
eint_spatial_std  = []

for plt_path in plotfiles:
    # Load the metadata from the metadata.yaml of the plotfile
    metadata_file = os.path.join(plt_path, "metadata.yaml")

    # Convert the yaml content in a python dictionary
    if os.path.exists(metadata_file):     # if the file exists, open it in reading mode
        with open (metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)  # convert the yaml content in a python dictionary

        # Retrive the cosmological parameters
        cosmo   = metadata.get('cosmology', {})   # look for the cosmology section; if not found, return an empty dict {}
        a_init  = cosmo.get('a_init')
        a_now   = cosmo.get('a')
        z_now   = cosmo.get('z')
        H0_cgs  = cosmo.get('H0')               # Hubble constant in cgs
        h       = cosmo.get('hubble_constant')
        omega_m = cosmo.get('Omega_m')
        omega_r = cosmo.get('Omega_r')
        omega_l = cosmo.get('Omega_Lambda')
        omega_k = cosmo.get('Omega_k')

        # Fill the dictionary of the scale factors
        a_values.append(a_now)
    else:
        print(f"Attention: metadata not found in {plt_path}")


    # Load the dataset for the field analysis
    ds = yt.load(plt_path)   # ds is the entire simulation dataset, containing grids, fields, and units.

    # print(ds.field_list) # to inspect the quanties

    # Create a data object that represents the entire simulation domain, with any field in the dataset
    ad = ds.all_data()  

    # Mean density 
    mean_rho = ad.quantities.weighted_average_quantity( # weighted average of density weighting each cell by its volume
    ("boxlib", "gasDensity"), ("index", "cell_volume"))

    # Mean internal energy
    mean_eint = ad.quantities.weighted_average_quantity(
    ("boxlib", "gasInternalEnergy"), ("index", "cell_volume"))

    # Mean momenta
    mean_xMom = ad.quantities.weighted_average_quantity(
    ("boxlib", "x-GasMomentum"), ("index", "cell_volume"))
    mean_yMom = ad.quantities.weighted_average_quantity(
    ("boxlib", "y-GasMomentum"), ("index", "cell_volume"))
    mean_zMom = ad.quantities.weighted_average_quantity(
    ("boxlib", "z-GasMomentum"), ("index", "cell_volume"))

    # Mean of the momentum magnitude (to check if numerical growth)
    mean_mom = np.sqrt(float(mean_xMom)**2 + float(mean_yMom)**2 + float(mean_zMom)**2)

    densities.append(float(mean_rho))     # float() since yt returns a unyt_quantity object, a number with the units
    energies.append(float(mean_eint))
    momenta.append(float(mean_mom))
    times.append(float(ds.current_time))  # imported directly from the simulation


    # Spatial standard deviations, to test the uniformity
    std_rho = ad.quantities.weighted_standard_deviation(
    ("boxlib", "gasDensity"), ("index", "cell_volume"))[0]         # [0] since returns (value, weight)

    std_eint = ad.quantities.weighted_standard_deviation(
    ("boxlib", "gasInternalEnergy"), ("index", "cell_volume"))[0]

    # Store the sd
    rho_spatial_std.append(float(std_rho))
    eint_spatial_std.append(float(std_eint))



print(f"Analyzed {len(a_values)} steps. Final a value: {a_values[-1]}")


# 2. Consistency check for Einstein-de Sitter between times and scale factors
a_values = np.array(a_values)                 # conversion in np array
times    = np.array(times)
t_start  = (2 / (3 * H0_cgs)) * (a_init**1.5) # cosmic time at the beginning of the simulation (s)

# 2a. Calculation of the scale factor
a_theoretical = (1.5 * H0_cgs * (t_start + times))**(2/3)

# 2b. Relative error
a_rel_err = (a_values - a_theoretical) / a_theoretical 
print(f"Relative error for a(t): {np.max(a_rel_err):.2e}")

# 2c. Fit
# Logarithms to do a linear fit
log_a = np.log10(a_values)
log_t = np.log10(t_start + times)

# Linear least squares polynomial fit, slope 
slope, intercept = np.polyfit(log_t, log_a, 1)   # 1: linear log(a) = intercept + slope * log(t)

print(f"Scale factor power law analysis:")
print(f"Theoretical exponent: {2 / 3}")
print(f"Fit of Quokka exponent: {slope:.4f}")
print(f"Difference: {np.abs(2 / 3 - slope):.2e}")

# 2d. Visualization
fig, ax = plt.subplots()
ax.plot(times, a_values, label="a(t)")

ax.set_title("Scale factor evolution: a(t) (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("t [s]", fontsize=11)
ax.set_ylabel("Scale factor (a)", fontsize=11)
ax.tick_params(direction='in', which='both')   # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', bbox_to_anchor=(0.25, 0.85))

# 2e. Insertion
ax_ins = inset_axes(ax, width='30%', height = '25%', loc='lower right', borderpad=3)  # 30% wide and 25% high of the main chart
ax_ins.plot(a_values, np.abs(a_rel_err), color='red', lw=1) # np.abs for log scale
ax_ins.set_title(r"Scale factor RE: $\frac{a_{\mathrm{sim}} - a_{\mathrm{pre}}}{a_{\mathrm{pre}}}$", fontsize=9)
ax_ins.tick_params(axis='both', labelsize=8)
ax_ins.grid(True, linestyle=':', alpha=0.5)
ax_ins.tick_params(direction='in', which='both')          # ticks point inward
ax_ins.axhline(0, color='black', lw=0.5, linestyle='--')  # reference line

# 2f. Text insertion in the figure
stats_text = (f'Fit slope: {slope:.4f} (expected: {2 / 3:.2f})')
ax.text(0.05, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_a_evolution.png', dpi=300)
print(f"Physical density plot saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")



# 3. Numerical Hubble parameter: check if the expansion is consistent, validation of the scale factor integrator

# 3a. Numerical H determination H = (da/dt) / a

# Arrays with the difference between a value and its left neighbour
da = np.diff(a_values)    
dt = np.diff(times)

dt_min = np.min(dt)
dt_max = np.max(dt)
print(f"Time step range: {dt_min:.2e} to {dt_max:.2e} s")

# Mean points for a 
a_mid = (a_values[1:] + a_values[:-1]) / 2.0

# Numerical H
H_num = da / (a_mid * dt)  # H = (da/dt) / a

# 3b. Theoretical H (EdS: H = H0 * a^-1.5)
H_theo = H0_cgs * (a_mid**(-1.5))

# 3c. Relative error
H_rel_err = (H_num - H_theo) / H_theo

print(f"\nHubble Parameter Analysis:")
print(f"  Mean H_rel_error: {np.mean(H_rel_err):.6e}")
print(f"  Max H_rel_error:  {np.max(H_rel_err):.6e}")

# 3d. Hubble parameter slope fit

# Logarithms to do a linear fit
log_a_mid = np.log10(a_mid)
log_H     = np.log10(H_num)

# Linear least squares polynomial fit, slope = -3(gamma - 1)
slope, intercept = np.polyfit(log_a_mid, log_H, 1)   # linear log(H) = intercept + slope * log(a)

print(f"Hubble parameter power law analysis:")
print(f"Theoretical exponent: {-1.5}")
print(f"Fit of Quokka exponent: {slope:.4f}")
print(f"Difference: {np.abs(-1.5 - slope):.2e}")


# 3e. Hubble parameter comparison plot
fig, ax = plt.subplots()
ax.plot(a_mid, H_num, 'o', ms=4, label="Quokka numerical $H(a)$", alpha=0.6)
ax.plot(a_mid, H_theo, label="Theoretical $H(a)$", ls='--', color='red')

ax.set_title("Hubble Parameter Validation (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"H(a) [$s^{-1}$]", fontsize=11)
ax.set_yscale('log')
ax.set_xlim(a_values[0], a_values[-1])
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend()

# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# For not long simulations
# Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax
# Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# Insertion of the Hubble parameter relative error
ax_ins = inset_axes(ax, width='30%', height = '25%', loc='right', borderpad=3)  # 30% wide and 25% high of the main chart
ax_ins.plot(a_mid, np.abs(H_rel_err), color='purple', lw=1, ms=3)  # np.abs for log scale
ax_ins.set_title("RE $(H_{num}-H_{th})/H_{th}$", fontsize=9)
ax_ins.tick_params(axis='both', labelsize=8)
ax_ins.grid(True, linestyle=':', alpha=0.5)
ax_ins.tick_params(direction='in', which='both')                   # ticks point inward
ax_ins.axhline(0, color='black', lw=0.5, linestyle='--')           # reference line
y_min, y_max = np.min(H_rel_err), np.max(H_rel_err)                # for the ylim
ax_ins.set_ylim(0, y_max * 1.2) 


# Text insertion in the figure
stats_text_H = (f'Fit slope: {slope:.4f} (expected: -1.5)')
ax.text(0.05, 0.95, stats_text_H,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_HubbleValidation.png', dpi=300)
print(f"Hubble parameter plot saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")



# 4. Comoving density
fig, ax = plt.subplots()
densities     = np.array(densities)         # to plot densities * 1e30
phys_dens     = densities / (a_values**3)   # rho_phys =  rho / a^3

ax.plot(a_values, densities * 10**30, label="Quokka Simulation", color='dodgerblue', lw=1.5, zorder=2)  # higher zorder comes before

# 4a. Reference rho = 1e-30 line
ax.axhline(1.0, color='red', linestyle='--', lw=2, label="Reference Value", zorder=1)

ax.set_title("Comoving density evolution (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"Density [$10^{-30} \cdot \mathrm{g/cm^3}$]", fontsize=11)
ax.yaxis.get_offset_text().set_visible(False)    # hid the exponential y upper left scale
ax.set_ylim(0.99, 1.01) # restrict the density range to capture possibly deviations 
ax.set_xlim(a_values[0], a_values[-1])
ax.tick_params(direction='in', which='both')   # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='upper left', frameon=True)

# 4b. Creation of a time x axis on the top
# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# 4c. Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # 4b. Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # 4c. Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# 4d. Spatial error subpanel
relative_error = (densities - densities[0]) / densities[0]

# Insertion creation
ax_ins = inset_axes(ax, width='30%', height = '25%', loc='lower right', borderpad=3)  # 30% wide and 25% high of the main chart
ax_ins.plot(a_values, relative_error, color='red', lw=1)
ax_ins.set_title("Relative Error", fontsize=9)
ax_ins.tick_params(axis='both', labelsize=8)
ax_ins.grid(True, linestyle=':', alpha=0.5)
ax_ins.tick_params(direction='in', which='both')          # ticks point inward
ax_ins.axhline(0, color='black', lw=0.5, linestyle='--')  # reference line

# 4e. Exact density and error norms
exact_densites = np.full_like(densities, densities[0])      # the comoving density remains the same

# Density norms
density_rel_err = np.abs((densities - exact_densites)) / exact_densites

# Norms of the relative error of the ensity of each snapshot (weighted average on the cells)
L1_rho   = np.mean(density_rel_err)
L2_rho   = np.sqrt(np.mean(density_rel_err**2))
Linf_rho = np.max(density_rel_err)

print(f"\nGlobal error norms (comoving density):")
print(f"  L1   = {L1_rho:.6e}")
print(f"  L2   = {L2_rho:.6e}")
print(f"  Linf = {Linf_rho:.6e}")

# Text insertion in the figure
stats_text_rho = (f'L1 Error: {L1_rho:.2e}\n'
              f'L2 Error: {L2_rho:.2e}\n'
              f'Linf Error: {Linf_rho:.2e}')
ax.text(0.60, 0.95, stats_text_rho,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# 4f. Save the figure
fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_DensityEvolution.png', dpi=300)
print(f"Comoving density saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs")


# 4g. Physical density
fig, ax = plt.subplots()
ax.plot(a_values, phys_dens, label="Quokka Physical Density ($\\rho_{\mathrm{com}} / a^3$)", lw =2, zorder=2)
ax.plot(a_values, densities, label="Quokka Comoving Density ($\\rho_{\mathrm{com}}$)", ls="--", lw=1.5, zorder=1)  # comoving density

ax.set_title("Physical vs comoving density (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"Density [$\mathrm{g/cm^3}$]", fontsize=11)
ax.set_xlim(a_values[0], a_values[-1])
ax.set_yscale('log')                           # physical density can vary of several orders of magnitudes
ax.tick_params(direction='in', which='both')   # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', frameon=True)

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_PhysDensity.png', dpi=300)
print(f"Physical density plot saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")



# 5. Internal energy

# 5a. Analytical solution and norms

# Analytical solution
def get_analytic_energy(e0, a_init, a_fin, gamma):
    return e0 * (a_init / a_fin)**(3 * (gamma - 1))

energies = np.array(energies)
gamma = 5 / 3    # monatomic ideal gas
analytical_energies = get_analytic_energy(energies[0], a_init, a_values, gamma)

# Norms
rel_err_array = np.abs((energies - analytical_energies) / analytical_energies)

# Norms of the relative error of the energy of each snapshot (weighted average on the cells)
L1_en   = np.mean(rel_err_array)                # arithmetic mean of errors
L2_en   = np.sqrt(np.mean(rel_err_array**2))    # square mean
Linf_en = np.max(rel_err_array)                 # worst relative error

# Print the norms
print(f"\nGlobal error norms (internal energy):")
print(f"  L1   = {L1_en:.6e}")
print(f"  L2   = {L2_en:.6e}")
print(f"  Linf = {Linf_en:.6e}")


# 5b. Fit of the energy slope

# Logarithms to do a linear fit
log_a = np.log10(a_values)
log_E = np.log10(energies)

# Linear least squares polynomial fit, slope = -3(gamma - 1)
slope, intercept = np.polyfit(log_a, log_E, 1)   # 1: linear log(E) = intercept + slope * log(a)

print(f"Internal energy power law analysis:")
print(f"Theoretical exponent: {-3 * (gamma - 1)}")
print(f"Fit of Quokka exponent: {slope:.4f}")
print(f"Difference: {np.abs(-3 * (gamma - 1) - slope):.2e}")

# 5c. Internal energy plot
fig, ax = plt.subplots()
ax.plot(a_values, energies, 'o', label="Quokka internal energy")
ax.plot(a_values, analytical_energies, label="Analytical solution", ls='-.')

ax.set_title("Internal Energy (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"Internal energy [$\mathrm{erg/cm^3}$]", fontsize=11)
ax.yaxis.get_offset_text().set_visible(False)    # hid the exponential y upper left scale
ax.set_xlim(a_values[0], a_values[-1])
ax.set_yscale('log')                             # energy can vary of several orders of magnitudes
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', bbox_to_anchor=(0.05, 0.95))

# Text insertion in the figure
stats_text = (f'Fit slope: {slope:.4f} (expected: {-3 * (gamma - 1):.1f})\n'
              f'L1 Error: {L1_en:.2e}\n'
              f'L2 Error: {L2_en:.2e}\n'
              f'Linf Error: {Linf_en:.2e}')
ax.text(0.5, 0.95, stats_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


# 5d. Creation of a time x axis on the top
# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# 5e. Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # 5d. Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # 5e. Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# 5f. Relative error
eint_rel_err = (energies - analytical_energies) / analytical_energies

# 5g. Insertion
ax_ins = inset_axes(ax, width='30%', height = '25%', loc='lower right', borderpad=3)  # 30% wide and 25% high of the main chart
ax_ins.plot(a_values, np.abs(eint_rel_err), color='red', lw=1) # np.abs for log scale
ax_ins.set_title("Relative Error", fontsize=9)
ax_ins.tick_params(axis='both', labelsize=8)
ax_ins.grid(True, linestyle=':', alpha=0.5)
ax_ins.tick_params(direction='in', which='both')          # ticks point inward
ax_ins.axhline(0, color='black', lw=0.5, linestyle='--')  # reference line

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_IntEnergy.png', dpi=300)
print(f"Energy saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")


# 5h. Energy error dedicated plot
fig, ax = plt.subplots()
ax.plot(a_values, eint_rel_err, label="Quokka internal energy relative error")

ax.set_title("Internal Energy Relative Error (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"RE$(E_{\mathrm{int}}) = \frac{E_{\mathrm{quokka}} - E_{\mathrm{pred}}}{E_\mathrm{{pred}}}$",  fontsize=11)
ax.yaxis.get_offset_text().set_visible(False)    # hid the exponential y upper left scale
ax.set_xlim(a_values[0], a_values[-1])
ax.set_yscale('symlog', linthresh=1e-6)          # symlog to account for positive and negative values,linthresh define the linear region to prevent the log diverge
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)

# Reference lines
ax.axhline(0, color='black', lw=0.5, ls='-')    # reference line on the 0
ax.axhline(1e-4, color='gray', lw=1, ls='--')   # tolerance lines of the test (1e-4)
ax.axhline(-1e-4, color='gray', lw=1, ls='--', label=r'Test tolerance: $10^{-4}$')
ax.legend(loc='best', bbox_to_anchor=(0.99, 0.95))

# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_IntEnergyError.png', dpi=300)
print(f"Energy saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")


# 6. Entropy level

# 6a. Norms

# Pressures of each sanpshot
phys_eint = energies / (a_values**3)      # physical internal energy 
pressures = phys_eint * (gamma - 1)       # ideal gas pressure from internal energy

# Specific entropy (expected constant for adiabatic expansion)
K = pressures / phys_dens**(gamma)

# Relative error from the costant initial value
K_rel_error = (K - K[0]) / K[0]

# Entropy norms
L1_K    = np.mean(np.abs(K_rel_error))         # arithmetic mean of errors
L2_K    = np.sqrt(np.mean(K_rel_error**2))     # square mean
Linf_K  = np.max(K_rel_error)                  # worst relative error

# Print the norms
print(f"\nGlobal error norms (specific entropy):")
print(f"  L1   = {L1_K:.6e}")
print(f"  L2   = {L2_K:.6e}")
print(f"  Linf = {Linf_K:.6e}")


# 6b. K plot
fig, ax = plt.subplots()
ax.plot(a_values, K, label="Specific entropy")
ax.axhline(K[0], color='black', lw=0.5, ls='-.', label="Reference value")    # reference line on the initial K value

ax.set_title("Specific entropy (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"Specific entropy [$P / \rho^{\mathrm{\gamma}}$]", fontsize=11)
ax.yaxis.get_offset_text().set_visible(True)    # hid the exponential y upper left scale
ax.set_xlim(a_values[0], a_values[-1])
ax.set_ylim(K[0] - K[0] * 1e-6, K[0] + K[0] * 1e-6) 
ax.set_yscale('linear')                          # linear scale useful for inspect a constant value
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', bbox_to_anchor=(0.95, 0.95))

# Text insertion in the figure
stats_text = (f'L1 Error: {L1_K:.2e}\n'
              f'L2 Error: {L2_K:.2e}\n'
              f'Linf Error: {Linf_K:.2e}')
ax.text(0.65, 0.25, stats_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_SpecEntropy.png', dpi=300)
print(f"Entropy saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")


# 7. Momentum: check if numerical growth (and potential Hubble drag check)
momenta    = np.array(momenta)          # moduli of the comoving momenta
velocities = momenta / densities        # peculiar velocities

# 7a. Analyical solution and norms
analytical_vel = velocities[0] * (a_values[0] / a_values)  # Hubble drag

print(f"\nPeculiar velocity analysis:")
print(f"  Initial Velocity: {velocities[0]:.2e} cm/s")
print(f"  Final Velocity:   {velocities[-1]:.2e} cm/s")

# Relative error
vel_rel_error = (velocities - analytical_vel) / analytical_vel

# Velocity norms
L1_vel   = np.mean(np.abs(vel_rel_error))         # arithmetic mean of errors
L2_vel   = np.sqrt(np.mean(vel_rel_error**2))     # square mean
Linf_vel = np.max(vel_rel_error)                  # worst relative error

# Print the norms
print(f"\nGlobal error norms (peculiar velocities):")
print(f"  L1   = {L1_vel:.6e}")
print(f"  L2   = {L2_vel:.6e}")
print(f"  Linf = {Linf_vel:.6e}")

# 7a. Velocity noise plot
fig, ax = plt.subplots()
ax.plot(a_values, velocities, 'o', label="Peculiar velocity")
ax.plot(a_values, analytical_vel, label=r"Analytical solution: $v_0 \cdot \frac{a_{\mathrm{in}}}{a}$")

ax.set_title("Peculiar Velocity Decay: Hubble Drag (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"Velocity [cm/s]", fontsize=11)
ax.yaxis.get_offset_text().set_visible(True)    # hid the exponential y upper left scale
ax.set_xlim(a_values[0], a_values[-1])
ax.set_yscale('log')                          # linear scale useful for inspect a constant value
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', bbox_to_anchor=(0.95, 0.95))

# Text insertion in the figure
stats_text_mom = (f'L1 Error: {L1_vel:.2e}\n'
                  f'L2 Error: {L2_vel:.2e}\n'
                  f'Linf Error: {Linf_vel:.2e}')
ax.text(0.65, 0.70, stats_text_mom,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Creation of a time x axis on the top
# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_VelocityNoise.png', dpi=300)
print(f"Velocity plot saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")


# 7b. Physical momentum (e.g. if added a non-vanishing momentum component)
phys_mom = momenta / (a_values**3)   # a^-3 since is a density of momentum

# 7c. Fit of the momentum slope

# Logarithms to do a linear fit
log_a     = np.log10(a_values)
log_p     = np.log10(phys_mom)
log_p_com = np.log10(momenta)

# Linear least squares polynomial fit, slope = -3(gamma - 1)
slope, intercept = np.polyfit(log_a, log_p, 1)   # 1: linear log(p) = intercept + slope * log(a)
slope_com_mom, intercept_com_mom = np.polyfit(log_a, log_p_com, 1)

print(f"Physical momentum density power law analysis:")
print(f"Theoretical exponent (Hubble drag + Volume dilution): -4.0") 
print(f"Fit of Quokka exponent: {slope:.4f}")
print(f"Difference: {np.abs(-4.0 - slope):.2e}")

print(f"Comoving momentum density power law analysis:")
print(f"Theoretical exponent (Hubble drag only): -1.0") 
print(f"Fit of Quokka exponent: {slope_com_mom:.4f}")
print(f"Difference: {np.abs(-1.0 - slope_com_mom):.2e}")

# Physical momentum plot
fig, ax = plt.subplots()
ax.plot(a_values, phys_mom, label="Quokka Physical Momentum ($p_{\mathrm{com}} / a^3$)", lw =2, zorder=1)
ax.plot(a_values, momenta, label="Quokka Comoving Momentum ($p_{\mathrm{com}}$)", ls="--", lw=1.5, zorder=2)  # comoving momentum

ax.set_title("Physical vs comoving momentum (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"Momentum [$\mathrm{g/cm^2 s}$]", fontsize=11)
ax.set_xlim(a_values[0], a_values[-1])
ax.set_yscale('log')                                   
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', frameon=True)

# Text insertion in the figure
stats_text = (f'Physical fit slope:    {slope:.4f} (expected: {-4.0})\n'
              f'Comoving fit slope: {slope_com_mom:.4f} (expected: {-1.0})')
ax.text(0.35, 0.75, stats_text,
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_PhysMom.png', dpi=300)
print(f"Physical momentum plot saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")


# 8. Omogeneity test: variation coefficient VC = sigma / mu

# 8a. Density
VC_density = np.array(rho_spatial_std) / densities   # scale the std for the current value

fig, ax = plt.subplots()
ax.plot(a_values, VC_density, label=r"$\sigma(\rho) / \rho $")

ax.set_title("Density Variation coefficient (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"$\sigma(\rho) / \rho $", fontsize=11)
ax.set_xlim(a_values[0], a_values[-1])
ax.set_yscale('linear')                          # linear scale useful for inspect a constant value
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', bbox_to_anchor=(0.95, 0.95))

# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_RhoSpatialSD.png', dpi=300)
print(f"Density VC plot saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")



# 8b. Internal Energy
VC_eint = np.array(eint_spatial_std) / energies   # scale the std for the current value

fig, ax = plt.subplots()
ax.plot(a_values, VC_eint, label=r"$\sigma(e_{\mathrm{int}}) / e_{\mathrm{int}}$")


ax.set_title("Internal Energy Variation coefficient (EdS)", pad=25, fontsize=14, fontweight='bold')
ax.set_xlabel("Scale factor (a)", fontsize=11)
ax.set_ylabel(r"$\sigma(e_{\mathrm{int}}) / e_{\mathrm{int}}$", fontsize=11)
ax.set_xlim(a_values[0], a_values[-1])
ax.set_yscale('linear')                          # linear scale useful for inspect a constant value
ax.tick_params(direction='in', which='both')     # ticks point inward
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='best', bbox_to_anchor=(0.95, 0.95))

# Target times for the upper ax (Gyr) and convert in seconds
target_times_Gyr = np.array([0, 50, 100, 150, 200, 294.5])
target_times_s   = target_times_Gyr * 1e9 * 3.15576e7   # the times array is in s

# Find the indices of the target times
indices = np.abs(times[:, None] - target_times_s).argmin(axis=0)  # times[:, None] convert times in matrix (N, 1), argimin for the index where the difference is min
indices = np.unique(indices)   # remove possible duplicate (if the simulation is short and two targets fall in the same timestep)

# Creation of a time x axis on the top
ax2 = ax.twiny()              # create the upper ax
ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# Map a correspondence between the upper and lower ax
ax2.set_xticks(a_values[indices])
ax2.set_xticklabels([f"{times[i]/(1e9 * 3.15576e7):.0f}" for i in indices])  # times in Gyr over the ticks
ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

# # Creation of a time x axis on the top
# ax2 = ax.twiny()              # create the upper ax
# ax2.set_xlim(ax.get_xlim())   # align with the lower ax

# # Map a correspondence between the upper and lower ax
# indices = np.linspace(0, len(a_values)-1, 5, dtype=int)   # 5 equispaced int indices chosen between all the plotfiles
# ax2.set_xticks([a_values[i] for i in indices])            # put the ticks in correspondence the the indices
# ax2.set_xticklabels([f"{times[i]/(3.154e7 * 1e9):.2f}" for i in indices])   # times in Gyr over the ticks
# ax2.set_xlabel("Time [Gyr]", fontsize=11, labelpad=10, loc="center")

fig.tight_layout()
fig.savefig('/data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/CosmoExp_EintSpatialSD.png', dpi=300)
print(f"Internal energy VC plot saved in /data/mfulghieri/quokka/my_worksite/myAnalysis/cosmological_expansion/outputs/")





# verificare per curiosità se i parametri di hubble e gli omega sono già parsabili
# con una simulazione test breve

# dopo tutti i run, fare tabelle di confronto di tutti i risultati

# verificare la differenza tra le norme amrex e quelle dell'analisi python

# Diagonal Test" e "Long-term Stress Test, aggiornare il max_timestep di blocco
# e lo stop_time di conseguenza.

# provare il test con budget diversi di omega. Modelli in cui posso confrontare con una soluzione
# analitic semplice e nota








