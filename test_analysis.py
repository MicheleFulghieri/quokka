import yt
import glob
import os
import sys
import numpy as np

os.chdir('/data/mfulghieri/quokka')
plts = sorted([d for d in glob.glob('plt00*') if os.path.isdir(d)])

if not plts:
    print("No plotfiles found in Quokka root.")
    sys.exit(0)

print(f"Found {len(plts)} plotfiles.")

for plt in plts:
    try:
        ds = yt.load(plt)
        ad = ds.all_data()
        
        # Check if CIC_density exists
        if ('deposit', 'CIC_density') in ds.derived_field_list:
            max_dens = ad['deposit', 'CIC_density'].max().v
            mean_dens = ad['deposit', 'CIC_density'].mean().v
            print(f"{plt}: max_CIC_density = {max_dens:.3e}, mean_CIC_density = {mean_dens:.3e}")
        else:
            print(f"{plt}: No CIC_density field")
    except Exception as e:
        print(f"Error loading {plt}: {e}")

