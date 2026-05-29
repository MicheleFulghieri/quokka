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

trans_to_save = np.column_stack((kh, trans_cdm, trans_bar, trans_tot))
np.savetxt(f'CAMB_lcdm_transfer_z{z_start}.dat', trans_to_save, 
           fmt='%12.6e',  # % min_fieldwidth number_of_digits exponential_notatation  
           header='k/h          CDM          Baryons      Total')

print(f"\nTransfert data saved in: {ic_path}\n")


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
