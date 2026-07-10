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
plotfiles_dir = "/data/mfulghieri/quokka/outputs/DMZeldovich/DMZeldovich_12622"
save_path = '/data/mfulghieri/quokka/my_worksite/myAnalysis/DMZeldovich/outputs'


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
a_values         = []
times            = []
delta_values     = []   # density contrast for each plotfile
delta_max_values = []   # max density contrast for each plotfile

# ----- Metadata retriving and History of the collapse -----
print("\n" + "-"*58)
print("Start analyzing the history of the collapse...\n")

# Initialize the figure for the history of the collapse
fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap('coolwarm_r')          # Colormap that gradients from blue to red


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

        # Fill the dictionary of the scale factors
        a_values.append(a_now)

        # ds is the entire simulation dataset, containing grids, fields, and units.
        times.append(float(ds.current_time))  # imported directly from the simulation

    else:
        print(f"Warning: metadata not found in {plt_path}")

    
    # Load the container of all the particles
    ad = ds.all_data()
    
    # ---- History of collapse ----

    # Creation of a line crossing the domain along x at the center (unyt_array with their unity of measure)
    c_vals = ds.domain_center.v            # domain_center returns [x_mid, y_mid, z_mid]
    left_vals = ds.domain_left_edge.v      # point where the line starts
    right_vals = ds.domain_right_edge.v    # point where the line ends
    
    start_point = [left_vals[0], c_vals[1], c_vals[2]]   # point where the line starts
    end_point   = [right_vals[0], c_vals[1], c_vals[2]]  # point where the line ends

    # Extraction of the data along the line
    ray = ds.ray(start_point, end_point)     # ray intersects the line with all 3D cells and particles in the simulation, extracting data only on that strip
   
    # Data sorting along x
    sort_idx = np.argsort(ray["index", "x"])            # index mask to growing sort the rays in x
    x_coord = ray["index", "x"][sort_idx].to('Mpc').v   # convert the coord from cm to Mpc and takes the admimensional value

    # Extract the density of DM deposited on the hydrodynamic grid with the CIC algoritm
    rho_dm  = ray[("deposit", "CIC_particles_density")][sort_idx].v

    # Local density contrast calculation
    rho_mean = np.mean(rho_dm)
    delta = (rho_dm / rho_mean) - 1.0
    delta_max = np.max(delta)

    delta_values.append(delta)
    delta_max_values.append(delta_max)

    # Physical values
    Lx = ad.ds.domain_width[0].to('Mpc').v 
    Ly = ad.ds.domain_width[1].to('Mpc').v 
    Lz = ad.ds.domain_width[2].to('Mpc').v 
    a_collapse = 0.5                              # a_collapse = 0.5 (as in the test)
    k          = 2 * np.pi / Lx                   # wave vector


    # Color map with color gradient over time
    color_val = cmap(i / len(plotfiles))

     # Plot   
    ax_hist.plot(x_coord, delta, color=color_val, label=f"a = {a_values[i]:.2f}")

    # If to many lines, plot only every two
    # if len(plotfiles) <= 2 or i % (len(plotfiles) // 2) == 0 or i == len(plotfiles) - 1:
    #     ax_hist.plot(x_coord, delta, label=f"a = {a_now:.3f}", color=color_val, linewidth=1.5)


    # ---- Theory-simulation comparison ----

    # Define a Lagrangian uniform grid q of the initial values 
    q = np.linspace(0, Lx, 1000)

    # Displacement amplitude of the particles: in EdS, the growing mode D(a) = a.
    # The initial amplitude was set as (a_init / a_collapse); at any time a, the amplitude is: A(a) = a / a_collapse.
    amplitude        = a_values[i] / a_collapse
    x_analytical     = q + (amplitude / k) * np.sin(k * q)    # perturbed position x = q - (A(a) / k) * sin(k * q)
    
    
    # Analytical density contrast (delta). In 1D Zel'dovich approximation, the density is:
    # rho(x) = rho_mean / |dx/dq| = rho_mean / |1 - A(a) * cos(k * q)|
    # Therefore: delta = (rho / rho_mean) - 1 = [1 / (1 - A(a) * cos(k * q))] - 1   
    delta_analytical = 1.0 / (1.0 + amplitude * np.cos(k * q)) - 1.0

    # Plotting Comparison
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))

    # Plot Numerical results (from the ray extraction)
    ax_comp.plot(x_coord, delta_values[i], label='Quokka (Numerical)', color='#1f77b4', linewidth=2, alpha=0.8)

    # Plot Analytical results
    ax_comp.plot(x_analytical, delta_analytical, label="Zel'dovich (Analytical)", 
             color='red', linestyle='--', linewidth=2)

    # Formatting
    ax_comp.set_title(f"Comparison: Numerical vs Analytical at $a = {a_values[i]:.2f}$", fontsize=14, fontweight='bold')
    ax_comp.set_xlabel("x [Mpc]", fontsize=11)
    ax_comp.set_ylabel(r"Density contrast $\delta$", fontsize=11)
    ax_comp.set_ylim(-1.5, max(delta) * 1.5)   # cut delta to the simulated one (the theoretical value breaks)
    ax_comp.set_xlim(x_coord[0], x_coord[-1])
    ax_comp.legend()
    ax_comp.grid(True, linestyle=':', alpha=0.6)

    fig_comp.tight_layout()
    fig_comp.savefig(os.path.join(save_path + "/DeltaComparison", f"CosmoDM_Analytical_Comparison_{i:03d}.png"), dpi=300)



    # --- Phase Space Analysis ---
    
    # Get position and velocities of all the particles
    p_x   = ad[("CIC_particles", "particle_position_x")].to('Mpc').v
    v_raw = ad[("CIC_particles", "particle_vx")].v    # since seems to be without units of measure, but Quokka works in cgs
    p_vx  = (v_raw * (cm / s)).to('km/s').v


    # Stats
    v_max_kms = np.max(np.abs(p_vx)) 
    print(f"Snapshot {i:2d} | a = {a_now:.3f} | z = {z_now:5.2f} | "
          f"Delta_max = {delta_max:6.2f} | v_max = {v_max_kms:7.2f} km/s")
    print(f"Total mass: {ad[('CIC_particles', 'particle_mass')].sum()}")



    # --- Creation of the plot for the phase space ---
    fig_ph, ax_ph = plt.subplots(figsize=(8, 5))
    ax_ph.scatter(p_x, p_vx, s=0.5, color=color_val, alpha=0.7)
    
    # Reference lines
    ax_ph.axhline(0, color='black', lw=1, alpha=0.3)
    ax_ph.axvline(ds.domain_center[0].to('Mpc').v, lw=0.5, color='black', alpha=0.3)

    # Formatting
    ax_ph.set_title(f"Phase Space Diagram - Step {i} ($a = {a_now:.3f}$)", fontweight='bold')
    ax_ph.set_xlabel("x [Mpc]")
    ax_ph.set_ylabel(r"$v_x$ (peculiar) [km/s]")
    ax_ph.set_xlim(x_coord[0], x_coord[-1])
    ax_ph.set_ylim(-300, 300)              # same physical limit for all the snap (in order to create a stable animation)
    ax_ph.grid(True, linestyle=':', alpha=0.5)
    
    # Save of each snapshot phase space plot
    fig_ph.tight_layout()
    fig_ph.savefig(os.path.join(save_path + "/PhaseSpace", f"PhaseSpace_{i:03d}.png"), dpi=150)
    plt.close(fig_ph)    # free the ram 


    # ---- Particle x-y and x-y-z positions ----
    x_cic = ad[("CIC_particles", "particle_position_x")].to('Mpc').v
    y_cic = ad[("CIC_particles", "particle_position_y")].to('Mpc').v
    z_cic = ad[("CIC_particles", "particle_position_z")].to('Mpc').v


    # --- Creation of the plot for the positions ---

    # Plot only a tenth of the particles
    n_sub = len(x_cic) // 10    # floor division (round to the integer)
    pos_sub = np.random.choice(len(x_cic), n_sub, replace=False)

    fig_pos, ax_pos = plt.subplots(figsize=(8, 5))
    ax_pos.scatter(x_cic[pos_sub], y_cic[pos_sub], s=0.5, alpha=0.7)

    
    # Formatting
    ax_pos.set_title(f"Particle Position - Step {i} ($a = {a_now:.3f}$)", fontweight='bold')
    ax_pos.set_xlabel("x [Mpc]")
    ax_pos.set_ylabel("y [Mpc]")
    ax_pos.set_xlim(0, Lx)
    ax_pos.set_ylim(0, Lz)      
    ax_pos.grid(True, linestyle=':', alpha=0.5)
    
    # Save of each snapshot x-y position plot
    fig_pos.tight_layout()
    fig_pos.savefig(os.path.join(save_path + "/ParticlePositions", f"X-Y_position_{i:03d}.png"), dpi=150)
    plt.close(fig_pos)    # free the ram 


    # 3D particle position scatter plot

    fig_3d_pos = plt.figure()
    ax_3d_pos = fig_3d_pos.add_subplot(projection='3d')

    # Plot only a fraction of the particles
    n_sub3d = len(x_cic) // 100    # floor division (round to the integer)
    pos_sub3d = np.random.choice(len(x_cic), n_sub3d, replace=False)

    ax_3d_pos.scatter3D(x_cic[pos_sub3d], y_cic[pos_sub3d], z_cic[pos_sub3d], s=0.01, alpha=0.9)
    ax_3d_pos.view_init(elev=20, azim=45)     # edge vision

    ax_3d_pos.set_xlabel('X [Mpc]')
    ax_3d_pos.set_ylabel('Y [Mpc]')
    ax_3d_pos.set_zlabel('Z [Mpc]')
    ax_3d_pos.set_xlim(0, Lx)
    ax_3d_pos.set_ylim(0, Ly)
    ax_3d_pos.set_zlim(0, Lz)


    fig_3d_pos.tight_layout()
    fig_3d_pos.savefig(os.path.join(save_path + "/ParticlePositions", f"3D_XYZ_position_{i:03d}.png"), dpi=150)
    plt.close(fig_3d_pos)    # free the ram 



    # ---- Hydro analysis ----

    # Density
    slc = yt.SlicePlot(ds, 'z', ('boxlib', 'gasDensity'))
    slc.annotate_quiver(('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), factor=12)
    slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
    slc.annotate_scale(corner="upper_right")

    slc.save(os.path.join(save_path + "/Hydro/Density", f"Density_{i:03d}.png"))

    # Internal energy
    slc = yt.SlicePlot(ds, 'z', ('boxlib', 'gasInternalEnergy'))
    slc.annotate_quiver(('boxlib', 'x-GasMomentum'), ('boxlib', 'y-GasMomentum'), factor=12)
    slc.annotate_timestamp(corner="upper_left", time=True, draw_inset_box=True)
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



# --- Back to the history of the collapse (to have it in single plot) ---
# Formatting
ax_hist.hlines(0, x_coord[0], x_coord[-1], label=r"$\delta = 0$", color='black', linestyle='--', alpha=0.5)
ax_hist.set_title("Zel'dovich Pancake: DM 1D Density Profile Evolution", pad=25, fontsize=12, fontweight='bold')
ax_hist.set_xlabel("x [Mpc]", fontsize=11)
ax_hist.set_ylabel(r"Density contrast: $\delta = \frac{\rho - \bar{\rho}}{\bar{\rho}}$", fontsize=11)
ax_hist.set_xlim(x_coord[0], x_coord[-1])
ax_hist.tick_params(direction='in', which='both')  # ticks point inward    
ax_hist.grid(True, which='both', linestyle=':', alpha=0.5)
ax_hist.legend(frameon=True, fontsize=9, loc='upper right', framealpha=0.9, ncol=2)

fig_hist.tight_layout(pad=1.5)
fig_hist.savefig(os.path.join(save_path, "CosmoDM_CollapseDensityDMHistory.png"), dpi=300)
print(f"Density DM collapse history saved in: {save_path}")


# Convert in numpy arrays times and scale factors
a_values         = np.array(a_values)
times            = np.array(times)
redshits         = (1 - a_values) / a_values    # z = (1 - a) / a 
delta_values     = np.array(delta_values)
delta_max_values = np.array(delta_max_values)









    # for field in ds.derived_field_list:
    #     if "gpot" in field:
    #         print("Success!")
    #     else:
    #         print("Fail")









# magari aggiungere una norma L1 o simile della differenza tra la soluzione analitica di 
# Zel'dovich e la delta di uno snapshot (magari non gli ultimissimi, dove la soluzione
# analitica esplode), interpolando i valori di quokka e applicando l'interpolatore ai valori
# esatti
# Quantitative Error Analysis: interpolate numerical solution onto the analytical grid for a direct comparison
# f_num = interp1d(x_coord, delta_values[-3], bounds_error=False, fill_value="extrapolate")  # interpolation of the discrete quokka solution (due to the resolution)
# delta_num_interp_3 = f_num(x_analytical)    # apply the interpolated quokka function to the analytical grid
# L1_error = np.mean(np.abs(delta_num_interp_3 - delta_analytical))
# print(f"L1 Error between Numerical and Analytical: {L1_error:.5e}")



# provare a ripetere l'analisi del test in c++ amrex in python per validarla






# for field in ds.derived_field_list:
#     if "gpot" in field:
#         print("Success!")
#     else:
#         print("Fail")





# La matematica dello spostamento spiega questo punto. La formula usata: 
# \(x = q - A \cos(k q)\).Il punto di fuga: A \(2.5\text{ Mpc}\) le particelle
#  si allontanano.Il punto di accumulo: A \(7.5\text{ Mpc}\) le particelle
#  convergono.L'effetto non lineare: L'evoluzione sposta il picco verso
#  \(7.8\text{ Mpc}\).Nel tuo box di \(10\text{ Mpc}\), la funzione coseno
#  spinge le particelle da destra e da sinistra verso il punto
#  \(3/4 \times 10 = 7.5\text{ Mpc}\).

# -sin nel dispalcement per il collasso al centro


# dimostrarsi matematicamente che il contrasto di densita propto a in EdS










# [('CIC_particles', 'particle_cpu'), ('CIC_particles', 'particle_id'), 
# ('CIC_particles', 'particle_mass'), ('CIC_particles', 'particle_position_x')
# , ('CIC_particles', 'particle_position_y'), ('CIC_particles', 'particle_position_z'), 
# ('CIC_particles', 'particle_vx'), ('CIC_particles', 'particle_vy'), 
# ('CIC_particles', 'particle_vz'), ('all', 'particle_cpu'), ('all', 'particle_id'), 
# ('all', 'particle_mass'), ('all', 'particle_position_x'), ('all', 'particle_position_y'),
#  ('all', 'particle_position_z'), ('all', 'particle_vx'), ('all', 'particle_vy'), ('all', 'particle_vz'), 
# ('boxlib', 'gasDensity'), ('boxlib', 'gasEnergy'), ('boxlib', 'gasInternalEnergy'), ('boxlib', 'x-GasMomentum'), 
# ('boxlib', 'y-GasMomentum'), ('boxlib', 'z-GasMomentum'), ('nbody', 'particle_cpu'), 
# ('nbody', 'particle_id'), ('nbody', 'particle_mass'), ('nbody', 'particle_position_x'), 
# ('nbody', 'particle_position_y'), ('nbody', 'particle_position_z'), ('nbody', 'particle_vx'), 
# ('nbody', 'particle_vy'), ('nbody', 'particle_vz')]



# Metodi e campi derivati
#   for field in ds.derived_field_list:
#         if "cic" in field[1] or "deposit" in field[0]:
#             print(field)
# Output:
# ('deposit', 'CIC_particles_cic')
# ('deposit', 'CIC_particles_count')
# ('deposit', 'CIC_particles_density')
# ('deposit', 'CIC_particles_mass')
# ('deposit', 'all_cic')
# ('deposit', 'all_count')
# ('deposit', 'all_density')
# ('deposit', 'all_mass')
# ('deposit', 'nbody_cic')
# ('deposit', 'nbody_count')
# ('deposit', 'nbody_density')
# ('deposit', 'nbody_mass')


# Per gestire i dati in modo efficace senza saturare la RAM 
# (specialmente con 262.144 particelle per centinaia di step), 
# devi cambiare strategia: non accumulare mai i dati grezzi di tutte 
# le particelle in una lista.



# vedere se utile importare altre quantità legate alle CIC per l'analisi

# gestire la conversione automatica delle unità di misura e eventuali valori comoventi
# o divisioni per h; controllare il file job_info o header: Dentro la cartella del plotfile,
#  ci sono file di testo che specificano che le lunghezze sono in cm (le unità base di Quokka).
# LIBRERIA unyt: yt usa internamente una libreria chiamata unyt. Quando scrivi .to('Mpc'), yt sa che 

# vedere se nell'analisi più significativo convertire le unità in km/s. Nel caso agire subito
# nella parte di estrazione dei dati


# Phase Space Diagram (vx vs x), density plot, spettro di potenza 1D, vedere 
# ultimo step, il collasso, Slicing 2D  della densità del gas (per vedere il picco, 
# anche se piccolo), studio dell'overdensity, creazione di un video per l'evoluzione, 
# creare una proiezione 2D "Face-on" e "Edge-on", efficace per vedere lo spessore del pancake,
#  Confrontare le posizioni estratte con la soluzione analitica dell'equazione di Zel'dovich.


# magari aggiungere un'anlisi sovrapposta di hydro e particle con i loro campi

# fare un test con densità gas simile a quella della DM, anche per controllare 
# convergenza hydro

# vedere se si riesce a fare un bella visualizzazione 3d e anche un'animazione




# Se per qualche analisi serve avere array con tutti i valori degli snaps:

# # --- Parameters ---

# # 1. Dictionaries to stores the evolution
# a_values  = []
# times     = []
# densities = []

# x      = []
# y      = []
# z      = []
# vx     = []
# vy     = []
# vz     = []
# masses = []


# for plt_path in plotfiles:
#     # Load the metadata from the metadata.yaml of the plotfile
#     metadata_file = os.path.join(plt_path, "metadata.yaml")

#     # Convert the yaml content in a python dictionary
#     if os.path.exists(metadata_file):     # if the file exists, open it in reading mode
#         with open (metadata_file, 'r') as f:
#             metadata = yaml.safe_load(f)  # convert the yaml content in a python dictionary

#         # Retrive the cosmological parameters
#         cosmo   = metadata.get('cosmology', {})   # look for the cosmology section; if not found, return an empty dict {}
#         a_init  = cosmo.get('a_init')
#         a_now   = cosmo.get('a')
#         z_now   = cosmo.get('z')
#         H0_cgs  = cosmo.get('H0')                  # Hubble constant in cgs
#         h       = cosmo.get('hubble_constant')
#         omega_m = cosmo.get('Omega_m')
#         omega_r = cosmo.get('Omega_r')
#         omega_l = cosmo.get('Omega_Lambda')
#         omega_k = cosmo.get('Omega_k')

#         # Fill the dictionary of the scale factors
#         a_values.append(a_now)
#     else:
#         print(f"Attention: metadata not found in {plt_path}")


#     # Load the dataset for the field analysis
#     ds = yt.load(plt_path)   # ds is the entire simulation dataset, containing grids, fields, and units.

#     # print(ds.field_list) # to inspect the quanties

#     # Create a data object that represents the entire simulation domain, with any field in the dataset
#     ad = ds.all_data()  

#     # ------- Hydro quantities --------

#     # Mean density 
#     mean_rho = ad.quantities.weighted_average_quantity( # weighted average of density weighting each cell by its volume
#     ("boxlib", "gasDensity"), ("index", "cell_volume"))

#     densities.append(float(mean_rho))     # float() since yt returns a unyt_quantity object, a number with the units


#     # ------- DM quantities ---------
   
#     # 1. Positions
#     CIC_x = ad['CIC_particles', 'particle_position_x'].to('Mpc').v  # 262144 (number of CICs) array
#     CIC_y = ad["CIC_particles", "particle_position_y"].to('Mpc').v
#     CIC_z = ad["CIC_particles", "particle_position_z"].to('Mpc').v

#     x.append(CIC_x)
#     y.append(CIC_y)
#     z.append(CIC_z)

#     # 2. Peculiar velocities (v_pec)
#     CIC_vx = ad["CIC_particles", "particle_vx"].v
#     CIC_vy = ad["CIC_particles", "particle_vy"].v
#     CIC_vz = ad["CIC_particles", "particle_vz"].v

#     vx.append(CIC_vx)
#     vy.append(CIC_vy)
#     vz.append(CIC_vz)

#     # 3. Mass (consistency check)
#     CIC_mass = ad["CIC_particles", "particle_mass"].to('g').v

#     masses.append(CIC_mass)


#     # ------ Simulation time ------
#     times.append(float(ds.current_time))  # imported directly from the simulation


# print(f"Analyzed {len(a_values)} steps. Final a value: {a_values[-1]}")


# # ----- Numpy array conversion ------
# a_values  = np.array(a_values)
# times     = np.array(times)
# densities = np.array(densities)

# x      = np.array(x)
# y      = np.array(y)
# z      = np.array(z)
# vx     = np.array(vx)
# vy     = np.array(vy)
# vz     = np.array(vz)
# # masses = np.array(masses)


#     # --- Retrive hydro quantities ----
    
#     # Hydro primitive
#     rho_gas  = ad[("boxlib", "gasDensity")]
#     eint_gas = ad[("boxlib", "gasInternalEnergy")]
#     momx_gas = ad[("boxlib", "x-GasMomentum")]
#     momy_gas = ad[("boxlib", "y-GasMomentum")]
#     momz_gas = ad[("boxlib", "z-GasMomentum")]

#     # Hydro derived
#     vx_gas    = momx_gas / rho_gas
#     vy_gas    = momy_gas / rho_gas
#     vz_gas    = momz_gas / rho_gas
#     v_mag_gas = np.sqrt(vx_gas * vy_gas + vy_gas * vy_gas + vz_gas * vz_gas)
#     gamma  = 5.0 / 3.0
#     press_gas = eint_gas / (gamma - 1)




# analizzare anche il potenziale gravitazionale aggiunto alle quantità salvate



# in QuokkaSimulation.hpp (riga 220), vedere se 
#  amrex::Real cosmology_dt_limit_ = PhysicsTraits<problem_t>::cosmology_dt_limit;
	# // a_half_ = a(t_n + dt/2): cached by particleCosmologyComputeHalfStep() before the hydro
	# // advance so that the drift and the Strang-split post-kick drag can both use the same
	# // bitwise-identical midpoint scale factor.
    
# è ancora necessario dopo l'uso del solutore nativo flessibile in
# Cosmology.hpp per l'equazione di Friedmann