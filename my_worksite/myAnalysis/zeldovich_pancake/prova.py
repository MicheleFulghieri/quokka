import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from unyt import Mpc, km, s, K, g, cm

# --- Configuration ---
h = 0.7
H0 = (h * 100.0 * km / s / Mpc).to('1/s')
a_init = 0.02
output_dir = "/data/mfulghieri/quokka/outputs/ZeldovichPancake_11237"
plotfiles = sorted(glob.glob(os.path.join(output_dir, "plt*")))

def get_a(t_sim):
    """Get the scale factor a(t) for Einstein-de Sitter."""
    t_init = (2.0/3.0) * (1.0/H0) * (a_init**1.5)
    t_abs = t_sim + t_init  # quokka start the time from 0, but the universe has already an age t(a_init)
    return (1.5 * H0 * t_abs)**(2.0/3.0)

if not plotfiles:
    print("Error: no plotfile found!")
    exit(1)

# Select some meaningful steps (e. g. start, half, end)
indices = [0, len(plotfiles)//2, len(plotfiles)-1]   # // rounds the division to the closest integer
selected_files = [plotfiles[i] for i in indices]

# Setup of the figures
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.05)

for plt_file in selected_files:
    ds = yt.load(plt_file)    # load the dataset
    a = get_a(ds.current_time).value
    print(ds.field_list)    
