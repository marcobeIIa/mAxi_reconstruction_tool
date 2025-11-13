import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
data_dir = Path(".")  # directory with the .npz files
output_file = data_dir / "EDE_rho_tot_plot.pdf"
max_points = 2000  # maximum points per sample to plot

# Load the data
data = np.load(data_dir / "all_backgrounds.npz", allow_pickle=True)
weights_data = np.load(data_dir / "all_weights.npz", allow_pickle=True)
weights = weights_data["weights"]

# Compute transparency from weights
w_min, w_max = np.min(weights), np.max(weights)
alphas = 0.1 + 0.9 * (weights - w_min) / (w_max - w_min + 1e-12)  # avoid div by zero

# --- Plot ---
plt.figure(figsize=(8,6))
for i in range(len(data["z"])):
    z = np.array(data["z"][i], dtype=np.float64)
    rho_tot = np.array(data["rho_tot"][i], dtype=np.float64)

    # Downsample if necessary
    if len(z) > max_points:
        step = len(z) // max_points
        z_plot = z[::step]
        rho_plot = rho_tot[::step]
    else:
        z_plot = z
        rho_plot = rho_tot

    plt.plot(
        np.log10(1 + z_plot),
        rho_plot,
        alpha=alphas[i],
        color='grey',
        label=f"Sample {i+1}"
    )

plt.xlabel(r"$\log_{10}(1+z)$", fontsize=14)
plt.ylabel(r"$\rho_\mathrm{tot}$", fontsize=14)
plt.title("Total EDE energy density over redshift", fontsize=16)
plt.xlim(1,6)
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=8)

# Save to PDF
plt.savefig(output_file)
print(f"✅ Plot saved to {output_file}")

