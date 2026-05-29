import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import camb
from camb import model, initialpower

print('Using CAMB %s installed at %s' % (camb.__version__, os.path.dirname(camb.__file__)))

# Useful paths
analysis_path = '/data/mfulghieri/quokka/my_worksite/myAnalysis/PowerSpectrumCosmo/outputs'
ic_path = '/data/mfulghieri/quokka/src/problems/CosmoPowerSpectrum/cosmo_ic'

# Parametrs
h = 0.675
Mpc_to_cm = 3.08567758e24
box_length_Mpc = 100.0
box_length = box_length_Mpc * Mpc_to_cm       # 100 Mpc in cm
ombh2 = 0.022                                 # barionic density (Omega_b * h^2)
omch2 = 0.122                                 # CMD density (Omega_c * h^2)
Omega_b = ombh2 / (h**2)
Omega_c = omch2 / (h**2)
Omega_m = Omega_b + Omega_c  
Omega_L = 1.0 - Omega_m                       # DE (flat universe)
sigma_8 = 0.811
z_start = 99.0

pars = camb.CAMBparams()                                          # istance the main camb class
pars.set_cosmology(H0=h*100, ombh2=ombh2, omch2=omch2, mnu=0.0)   # background parameters
pars.InitPower.set_params(ns=0.965, As=2e-9)                      # inflaction parameters
pars.set_matter_power(redshifts=[z_start], kmax=100.0)        # high kmax -> resolve small scales
pars.NonLinear = model.NonLinear_none                         # force linear spectrum (in view of MUSIC)

results = camb.get_results(pars)   # solve via the undelying Fortran the Boltzmann equation from the inflaction to z_start

# Istance of the tranfert results
trans = results.get_matter_transfer_data()   # returns the quantities divided by k^2
kh = trans.transfer_data[0, :, 0]            # (222,) array values of k/h

# [camb.model.desired_variable, range of values, redshift]
trans_cdm = trans.transfer_data[model.Transfer_cdm - 1, :, 0]  # k/h, 0 for z=z_start (the only present)
trans_bar = trans.transfer_data[model.Transfer_b - 1, : , 0]   # ordinary matter
trans_tot = trans.transfer_data[model.Transfer_tot - 1, :, 0]  # DM + ordinary matter + nu



# ---- Transfert function inspection ----
fig_tf, ax_tf = plt.subplots()

ax_tf.plot(kh, trans_cdm, label='CDM', color='red', linestyle='--')
ax_tf.plot(kh, trans_bar, label='Baryons (Gas)', color='green', linestyle=':')
ax_tf.plot(kh, trans_tot, label='Total Matter', color='black', alpha=0.7)

ax_tf.set_xscale('log')
ax_tf.set_yscale('log')
ax_tf.set_xlabel(r'$k \; (Mpc/h)^{-1}$')
ax_tf.set_ylabel(r'$T(k) \; (Mpc/h)^{2}$')
ax_tf.set_title(f'Transfer function comparison at z={z_start}')
ax_tf.legend()
ax_tf.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(analysis_path, f"CAMB_TFplot_z_{z_start}.png"), dpi=300)
plt.close()


# ---- Native Power spectrum inspection ----

# Extract the spectrum as arrays (Mpc with h units, already MUSIC optimized)
kh, z, pk = results.get_matter_power_spectrum(
    minkh=1e-4,      # large scale max
    maxkh=100.0,     # small scale min
    npoints = 1000   # number of sampled points
)

# Formattation and exportation to MUSIC
PS_to_save = np.column_stack((kh, pk[0,:]))   # stack k and pk (with the only z=z_start) in two columns: npoints rows [(kh_i, ph_i)] for i=0,..., npoints
np.savetxt(f'CAMB_PS_check_z{z_start}.txt', PS_to_save, header='k/h (Mpc^-1) | P(k) (Mpc/h)^3')

# PS function
fig, ax = plt.subplots()
ax.plot(PS_to_save[:,0], PS_to_save[:,1], label='Power spectrum')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k (Mpc/h)^{-1}$')
ax.set_ylabel(r'$P(k) (Mpc/h)^{3}$')
ax.set_title(f'LCDM matter power spectrum at z={z}')
ax.legend()
ax.grid(True, alpha=0.5)


plt.tight_layout()
plt.savefig(os.path.join(analysis_path, f"CAMB_PSplot_z_{z_start}.png"), dpi=300)
plt.close()
print(f"\nPower spectrum plot saved in: {analysis_path}")


# ---- Power spectrum from transfert function comparison ----

# PS from the tranfert function
kh_native = trans.transfer_data[0, :, 0]  
k_physical = kh_native * results.Params.h   # k = (k/h) * h
primordial_PK = results.Params.scalar_power(k_physical)
matter_power = primordial_PK * trans_tot**2 * k_physical**4 / (k_physical**3 / (2 * np.pi**2))




# Native PS
kh2, zs, PK = results.get_linear_matter_power_spectrum(hubble_units=False)

# Graphical comparison
fig_comp, ax_comp = plt.subplots()
ax_comp.plot(k_physical, matter_power, label='Power spectrum from Transfert func')
ax_comp.plot(kh, PK[0, :], label='CAMB native Power spectrum')
ax_comp.set_xscale('log')
ax_comp.set_yscale('log')
ax_comp.set_xlabel(r'$k\, [h Mpc^{-1}]$')
ax_comp.set_ylabel(r'$P(k) (Mpc/h)^{3}$')
ax_comp.set_title(f'Comparison CAMB matter power spectrum at z={z}')
ax_comp.legend()
ax_comp.grid(True, alpha=0.5)


plt.tight_layout()
plt.savefig(os.path.join(analysis_path, f"CAMB_PScheck_z_{z_start}.png"), dpi=300)
plt.close()
print(f"\nComparison power spectrum plot saved in: {analysis_path}")


# Input values for MUSIC and quokka
print(f"-" * 75)
print(f"VALUES FOR MUSIC AND QUOKKA")
print(f"Hubble parameter:  {h}")
print(f"Box length:        {box_length_Mpc} Mpc | {box_length} cm")
print(f"Box length/h :     {box_length_Mpc / h} Mpc | {box_length / h} cm")
print(f"Omega_m = {Omega_m:.6f}")
print(f"Omega_b = {Omega_b:.6f}")
print(f"Omega_L = {Omega_L:.6f}")
print(f"-" * 75)


# sistemare ultimo plot di confronto