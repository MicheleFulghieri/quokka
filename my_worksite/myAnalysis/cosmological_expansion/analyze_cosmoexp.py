import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml
from natsort import natsorted
from unyt import Mpc, km, s, g, cm



# --- Path to plotfiles ---
output_dir = "/data/mfulghieri/quokka/outputs/CosmologicalExpansion_11717"

# retrive the plotfiles 
plotfiles = glob.glob(os.path.join(output_dir, "plt*"))
plotfiles = natsorted(plotfiles)    

if not plotfiles: 
    print("No plotfiles found!")
    exit(1)



# --- Parameters ---
for plt in plotfiles:
    ds = yt.load(plt)


def get_a_factor(plt_path):
    with open(f"{plt_path}/metadata.yaml", 'r') as f:
        meta = yaml.safe_load(f)
    return meta.get('a')

a_now = get_a_factor("/data/mfulghieri/quokka/outputs/CosmologicalExpansion_11717/plt0000000")
print(f"Scale factor: {a_now}")


with open("/data/mfulghieri/quokka/outputs/CosmologicalExpansion_11717/plt0000000/metadata.yaml") as f:
    meta = yaml.safe_load(f)
print(meta["cosmology"])




# find a way to import h, omega_m, ... directly from the input file or from the plotfiles
#?????h = yt_.load(plotfiles[0]).parameters['h']


