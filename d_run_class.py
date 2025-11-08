import json
import numpy as np
from classy import Class

with open("params_mEDE.json") as f:
    params_mEDE = json.load(f)

EDEm = Class()
EDEm.set(params_mEDE)
#EDEm.compute()

bg_m = EDEm.get_background()

print(bg_m.keys())
derived = EDEm.get_current_derived_parameters(['z_eq','z_rec'])


z = bg_m['z']
rho_tot = np.zeros_like(z)
f_EDE_tot = np.zeros_like(z)

#import matplotlib.pyplot as plt
# 
# fig, ax = plt.subplots()
# N = int(params_mEDE['N_mscf'])
# for i in range(N):  # Corrected to use params_mEDE
#     rho_EDE_m = bg_m[f'(.)rho_mscf[{i}]']
#     f_EDE_m = rho_EDE_m / bg_m['(.)rho_tot']
#     ax.plot(1+z, f_EDE_m, label=f'$f_{{EDE}}$ (i={i})')
#     f_EDE_tot += f_EDE_m
#     
# ax.plot(1+z, f_EDE_tot, label=f'$f_{{tot}}$')
# ax.set_xlabel(r'$1+z$',fontsize='large')
# ax.set_ylabel(r'$f_{EDE}$',fontsize='large')
# #ax.set_ylim(0, .1)  # Set based on physical expectations
# #ax.tick_params(colors='darkred')
# 
# lines, labels = ax.get_legend_handles_labels()
# ax.legend(lines, labels)
# ax.set_xscale('log')
# ax.set_yscale('linear')
# ax.set_xlim(1,1e5)
# #ax.set_ylim(1e-7,5e-1)
# 
# plt.title("mAxiCLASS: Fraction of EDE in each field (theory parameters)")
# plt.tight_layout()
# plt.show()




# Collect all mscf components and sum them
N = int(params_mEDE['N_mscf'])
rho_components = []
for i in range(N):
    rho_i = bg_m[f'(.)rho_mscf[{i}]']/bg_m['(.)rho_tot'] #now we are dealing with energy fractions
    rho_components.append(rho_i)
    rho_tot += rho_i
    rho_tot = rho_tot
# --- Append results to global storage (for multi-run usage) ---
try:
    # If file already exists, append to it
    existing = np.load("all_backgrounds.npz", allow_pickle=True)
    all_z = list(existing["z"])
    all_rho_tot = list(existing["rho_tot"])
    all_rho_components = list(existing["rho_components"]) if "rho_components" in existing else []
except FileNotFoundError:
    all_z, all_rho_tot, all_rho_components = [], [], []

# Append current sample
all_z.append(z)
all_rho_tot.append(rho_tot)
all_rho_components.append(np.array(rho_components))

# Save everything again
np.savez("all_backgrounds.npz",
         z=np.array(all_z, dtype=object),
         rho_tot=np.array(all_rho_tot, dtype=object),
         #rho_components=np.array(all_rho_components, dtype=object))
         )
