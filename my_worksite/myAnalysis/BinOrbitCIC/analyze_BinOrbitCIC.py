# note: requires yt>=4.3.0
import yt
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Anti-Grain Geometry, backend without interface, allow to save the plots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # for the color map
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob
import yaml
from natsort import natsorted
from unyt import Mpc, km, s, g, cm


# Save path
save_path = '/data/mfulghieri/quokka/my_worksite/myAnalysis/BinOrbitCIC/outputs'

# --- Path to plotfiles ---
output_dir = "/data/mfulghieri/quokka/outputs/BinOrbitCIC/BinOrbitCIC_12039"


# retrive the plotfiles 
# os.path.join joins the folder path with the "plt*" pattern, glob.glob creates a list of all files/folders that begin with "plt"
plotfiles = glob.glob(os.path.join(output_dir, "plt*"))
#plotfiles = [f for f in glob.glob(os.path.join(output_dir, "plt[0-9]*")) if ".old" not in f]
plotfiles = natsorted(plotfiles)     # natural ordering to the list 

if not plotfiles: 
    print("No plotfiles found!")
    exit(1)


# Set yt's verbosity 
yt.set_log_level(40)   # level 40 only displays critical errors


# Function for the error introduced by the simulation
def get_particle_dist(plotfiles):

    # Lists initialization
    t_arr       = []
    err_arr     = []
    err_vel_arr = []
    pos         = []
    nx_frame0   = None

    # Theoretical physical paramters
    d0 = 2.0 * 3.125e12  # distance (cm)
    v0 = 10332860.       # orbital velocity
    m0 = 2.0e34          # stellar mass

    # Load the plotfiles
    for pltfile in plotfiles:

        ds = yt.load(pltfile)
        # print(ds.derived_field_list)
        Lx = ds.domain_right_edge[0] - ds.domain_left_edge[0]   # domain length
        Nx = ds.domain_dimensions[0]     # number of cells (x)
        cell_dx = Lx/Nx                  # cell dimension

        # Extraction of the physical quantities
        ad = ds.all_data()     # all data container 

        x = ad["CIC_particles", "particle_position_x"]   # ad returns a unyt array (dimensional number)
        y = ad["CIC_particles", "particle_position_y"]
        z = ad["CIC_particles", "particle_position_z"]
        pos.append((float(x[0].value), float(y[0].value), float(z[0].value)))

        vxs = ad["CIC_particles", "particle_vx"]
        vys = ad["CIC_particles", "particle_vy"]
        vzs = ad["CIC_particles", "particle_vz"]
        ms = ad["CIC_particles", "particle_mass"]

        # Verify that the mass of the star remained 2.0e34
        assert np.isclose(ms[0], m0) and np.isclose(ms[1], m0) 

        # Realtive distance between the stars (0 star 1 and 2 star 1)  
        dx = x[0] - x[1]
        dy = y[0] - y[1]
        dz = z[0] - z[1]
        d = np.sqrt(dx*dx + dy*dy + dz*dz)   # euclidean distance (Pitagora)
        #fractional_err = (d-d0)/d0
        #grid_err = (d - d0) / cell_dx.value
        grid_err = (d.value - d0) / cell_dx.value  # relative error with respect to theory

        # Velociy and velocity error for the first star
        vx = vxs[0]
        vy = vys[0]
        vz = vzs[0]
        v_mag = np.sqrt(vx*vx + vy*vy + vz*vz)
        # err_vel = (v_mag - v0) / v0
        err_vel = (v_mag.value - v0) / v0

        t_arr.append(float(ds.current_time) / 3.15e7)  # convert simulation time from seconds to yrs
        
        err_arr.append(grid_err)
        err_vel_arr.append(err_vel)

        # Keep track of resolution (can change with AMR)
        if nx_frame0 is None:    
            nx_frame0 = Nx

    return t_arr, err_arr, err_vel_arr, pos, nx_frame0


# Track the particle postions
def get_particle_orbit(plotfiles):

    # List initialization (now keep separate the stars positions)
    t_arr    = []
    pos1_arr = []
    pos2_arr = []


    for pltfile in plotfiles:
        ds = yt.load(pltfile)
        ad = ds.all_data()

        # Load the positions of all the particles in the simulation (here 2 -> x, y, z have dim=2)
        x = ad["CIC_particles", "particle_position_x"]
        y = ad["CIC_particles", "particle_position_y"]
        z = ad["CIC_particles", "particle_position_z"]

        t_arr.append(float(ds.current_time) / 3.15e7)

        # Create 2 separate tuples for (x,y,z) of each star
        pos1_arr.append((float(x[0].value), float(y[0].value), float(z[0].value)))
        pos2_arr.append((float(x[1].value), float(y[1].value), float(z[1].value)))

    return t_arr, pos1_arr, pos2_arr   # poisitions of the two stars for each time


# Function to coordinate the previous two
def plot_orbit_and_error(pltdir):

    # Create the list of all the plotfile snapshots
    files = glob.glob(pltdir + "/plt*")   # Create a list of all paths starting with "plt" inside the pltdir folder
    files = sorted(files)                 # time order the plotfiles

    # Call the function of the simulation errors passing the list of the plotfiles
    t, err_dist, err_vel, pos_particle0, nx_frame0 = get_particle_dist(files)

    # Distance and orbital velocity error
    print("max rel_error distance: {:.1e}".format(np.max(np.abs(err_dist))))
    print("max error velocity: {:.1e}".format(np.max(np.abs(err_vel))))

    # Create a terminal table to monitor the evolution of the speed error step by step.
    print("time(yr) rel_err_vel within_tol_1e-3 within_tol_1e-2?")
    for i in range(len(t)): # print time vs err_vel as a table
        print("{:.1e} {:.1e} {} {}".format(t[i], err_vel[i], np.abs(err_vel[i]) < 1.0e-3, np.abs(err_vel[i]) < 1.0e-2))

    # Plot the orbit
    # Call the position function
    t, pos1_arr, pos2_arr = get_particle_orbit(files)

    # Orbit scatter plot
    plt.figure(figsize=(6,6))
    Nx = nx_frame0
    plt.title(f"nx={Nx}")
    # List comprension to extracts only the X coordinates from all saved tuples to create the horizontal axis
    plt.scatter([pos[0] for pos in pos1_arr], [pos[1] for pos in pos1_arr], color="red", label="particle 1")
    plt.scatter([pos[0] for pos in pos2_arr], [pos[1] for pos in pos2_arr], color="blue", label="particle 2")
    plt.legend()
    ax = plt.gca()
    ax.set(xlabel="x", ylabel="y")
    ax.axis("equal")
    ax.grid()
    hw = 6.0e12
    ax.set(xlim=(-hw, hw), ylim=(-hw, hw))
    figdir = save_path
    #figdir = os.path.dirname(files[0])   #files[0] takes the path of the first plotfile, and os.path.dirname extracts the folder containing it
    plt.savefig(os.path.join(figdir, "orbit.png"), dpi=300)


    # Error plot
    plt.figure(figsize=(6,4))
    plt.plot(t[1:], np.abs(err_dist[1:]))
    # plt.ylim(-0.1, 0.1)
    plt.grid()
    plt.xlabel("time (yr)")
    plt.ylabel(r"$(d-d_0)/\Delta x$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "orbit_err.png"), dpi=300)

    return


# Deep diagnostics of the rhs of Poisson equation
def plot_debug_Poisson_rhs(pltdir):


    # Quokka can generate special files called debug_... when the gravity solver encounters problems. 
    debug_files = sorted(glob.glob(pltdir + "/debug_*"))

    # A. No debug file in the folder
    if not debug_files:
        print(f"--- INFO: No debug file found in {pltdir} ---")
        return
    # Skip if there is not 
    for pltfile in debug_files:
        if not os.path.isdir(pltfile):
            continue
            
         # B. There are dedug files, but not for acceleration
        is_accel = "accel" in os.path.basename(pltfile)
        if not is_accel:
            print(f"--- Skip {os.path.basename(pltfile)}: no acceleration debug ---")
            continue
        
        # C. Acceleration debug file
        print(f"--- Generating debug for: {os.path.basename(pltfile)}")


        # Sclice along z (xy plane)
        ds = yt.load(pltfile)
        field = ("boxlib", "accel_x")       # mark with different colours different accelerations
        slc = yt.SlicePlot(ds, "z", field)

        # Log scale for the colours
        if is_accel:
            slc.set_zlim(field, 1e0, 3e4)
            slc.set_log(field, True)

        # Draw the edges of AMR blocks and individual cells.
        # Allow to see if the gravity solver is becoming inaccurate right where the grid resolution changes (at the boundary between layers)
        slc.set_width((1.3e13, "cm"))
        slc.annotate_grids()   
        slc.annotate_cell_edges(line_width=0.001, color='black')

        # Name the image file using the same name as the debug folder, adding the extension.
        figfn = pltfile
        if figfn[-1] == '/':
            figfn = figfn[:-1]
        figfn = figfn + "_accel_x.png"
        slc.save(figfn, mpl_kwargs={"dpi": 300})


if __name__ == "__main__":

    # Default if no arguments passed via terminal
    pltdir = output_dir

    # Use the terminal input if passed
    if len(sys.argv) > 1:
        pltdir = sys.argv[1]

    print(f"Analysing folder: {pltdir}")

    plot_orbit_and_error(pltdir)
    plot_debug_Poisson_rhs(pltdir)



# To the input file for poisson debug
# --- Parametri di Debug per la Gravità ---
# gravity.debug = 1                     # Attiva la modalità debug per la gravità
# gravity.save_rhss = 1                 # Salva il lato destro (massa spalmata)
# gravity.save_accel = 1                # Salva i campi di accelerazione calcolati
# gravity.save_phi = 1                  # Salva il potenziale gravitazionale grezzo