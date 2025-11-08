import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load("all_backgrounds.npz", allow_pickle=True)
# load the weight data
weights_data = np.load("all_weights.npz", allow_pickle=True)
weights = weights_data["weights"]

w_min, w_max = np.min(weights), np.max(weights)
alphas = 0.1 + 0.9 * (weights - w_min) / (w_max - w_min + 1e-12)  # avoid div by zero
#weight = data["weight"]

# Plot each sample
plt.figure(figsize=(8,6))
for i in range(len(data["z"])):
    # Normalize weights to [0.1, 1] for plotting transparency
    z = np.array(data["z"][i], dtype=np.float64) 
    rho_tot = np.array(data["rho_tot"][i], dtype=np.float64)
    #plt.plot(np.log10(1 + z), rho_tot, label=f"Sample {i+1}")
    weight = weights[i]  # <--- use it for scaling / labeling
    plt.plot(
        np.log10(1+z), 
        rho_tot, 
        label=f"Sample {i+1}", 
        alpha=alphas[i],  # set transparency proportional to weight
        color='grey',
    )
plt.xlabel(r"$\log_{10}(1+z)$", fontsize=14)
#plt.xlim(1,6)
#plt.ylim(0,0.1)
plt.ylabel(r"$\rho_\mathrm{tot}$", fontsize=14)
plt.title("Total EDE energy density over redshift", fontsize=16)
#plt.legend()
plt.grid(True)
plt.show()

# Show the plot
plt.tight_layout()

