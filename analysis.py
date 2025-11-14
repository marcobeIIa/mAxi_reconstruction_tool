import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your fresh run data
data_dir = Path("/Users/bellamarco01/uni/1_master_thesis/tools/output_remote/")
max_points = 200  # downsampling if necessary

# Load the data
data = np.load(data_dir / "all_backgrounds.npz", allow_pickle=True)
weights_data = np.load(data_dir / "all_weights.npz", allow_pickle=True)
weights = weights_data["weights"]

# Compute alpha (transparency) from weights
w_min, w_max = np.min(weights), np.max(weights)
alphas = 0.1 + 0.9 * (weights - w_min) / (w_max - w_min + 1e-12)

# Prepare arrays for statistics
z_list = []
rho_list = []

for i in range(len(data["z"])):
    z = np.array(data["z"][i], dtype=np.float64)
    rho = np.array(data["rho_tot"][i], dtype=np.float64)

    # Downsample to max_points if necessary
    if len(z) > max_points:
        step = len(z) // max_points
        z = z[::step]
        rho = rho[::step]

    z_list.append(z)
    rho_list.append(rho)

# Stack into 2D arrays: shape = (Nsamples, Npts)
Z = np.stack(z_list)
RHO = np.stack(rho_list)

# Compute statistics
median_rho = np.median(RHO, axis=0)
lower_68 = np.percentile(RHO, 16, axis=0)
upper_68 = np.percentile(RHO, 84, axis=0)
lower_95 = np.percentile(RHO, 2.5, axis=0)
upper_95 = np.percentile(RHO, 97.5, axis=0)

# Plot
plt.figure(figsize=(8,6))

# 95% confidence interval
plt.fill_between(np.log10(1 + Z[0]), lower_95, upper_95, color='lightgrey', label="95% CL")

# 68% confidence interval
plt.fill_between(np.log10(1 + Z[0]), lower_68, upper_68, color='darkgrey', label="68% CL")

# Median line
plt.plot(np.log10(1 + Z[0]), median_rho, color='black', lw=2, label="Median")

# Optional: overlay individual weighted samples
for i in range(Z.shape[0]):
    plt.plot(np.log10(1 + Z[i]), RHO[i], color='lightblue', alpha=0.01*alphas[i])

plt.xlabel(r"$\log_{10}(1+z)$", fontsize=14)
plt.ylabel(r"$\rho_\mathrm{tot}$", fontsize=14)
plt.yscale('log')
plt.title("Total EDE Energy Density — Confidence Levels", fontsize=16)
plt.xlim(1,5)
#plt.ylim(1e-3,1e5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

