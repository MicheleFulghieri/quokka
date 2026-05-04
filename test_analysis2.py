import yt
import sys
import numpy as np

for plt in ['plt0000000', 'plt0000395']:
    try:
        ds = yt.load(plt)
        ad = ds.all_data()
        print(f"--- {plt} ---")
        if ('deposit', 'CIC_density') in ds.derived_field_list:
            max_dens = ad['deposit', 'CIC_density'].max().v
            mean_dens = ad['deposit', 'CIC_density'].mean().v
            print(f"CIC_density: max = {max_dens:.3e}, mean = {mean_dens:.3e}")
        else:
            print("No CIC_density")
    except Exception as e:
        print(f"Error {plt}: {e}")
