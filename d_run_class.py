import json
import numpy as np
from classy import Class

with open("params_mEDE.json") as f:
    params_mEDE = json.load(f)

EDEm = Class()
EDEm.set(params_mEDE)
#EDEm.compute()

bg_m = EDEm.get_background()

derived = EDEm.get_current_derived_parameters(['z_eq','z_rec'])


z = bg_m['z'] #redshift
H = bg_m['H [1/Mpc]'] #hubble rate
rho_tot = np.zeros_like(z) #total mscf energy
#f_EDE_tot = np.zeros_like(z)

# Collect all mscf components and sum them
N = int(params_mEDE['N_mscf'])
rho_components = []
for i in range(N):
    rho_i = bg_m[f'(.)rho_mscf[{i}]']/bg_m['(.)rho_tot'] #now we are dealing with energy fractions
#    rho_i = bg_m[f'(.)rho_mscf[{i}]']
#    rho_components.append(rho_i)
    rho_tot += rho_i
    rho_tot = rho_tot
# --- Append results to global storage (for multi-run usage) ---
try:
    # If file already exists, append to it
    existing = np.load("all_backgrounds.npz", allow_pickle=True)
    all_z       = list(existing["z"])
    all_rho_tot = list(existing["rho_tot"])
    all_H       = list(existing["H"]) if "H" in existing else []
    #all_rho_components = list(existing["rho_components"]) if "rho_components" in existing else []
except FileNotFoundError:
#    all_z, all_rho_tot, all_rho_components = [], [], []
    all_z, all_rho_tot, all_H  = [], [], []

# --------------------------------------------------
# === DOWNSAMPLE BEFORE SAVING ===
# --------------------------------------------------

NPTS = 200   # choose 100, 200, 300… depending on how small you want the file

idx = np.round(np.linspace(0, len(z) - 1, NPTS)).astype(int)

z_small       = z[idx]
rho_tot_small = rho_tot[idx]
H_small       = H[idx]

# Append current sample
all_z.append(z_small)
all_rho_tot.append(rho_tot_small)
all_H.append(H_small)
#all_rho_components.append(np.array(rho_components))

# Save everything again
np.savez("all_backgrounds.npz",
         z=np.array(all_z, dtype=object),
         rho_tot=np.array(all_rho_tot, dtype=object),
         H=np.array(all_H, dtype=object),     
         #rho_components=np.array(all_rho_components, dtype=object))
         )

# clean up
EDEm.empty()
del EDEm
print("cleaned the cosmology!")
