import os
from pathlib import Path

# --- Import your local modules ---
from a_fixed_parameters_to_ini import main as make_fixed_ini
from b_generate_ini_from_chain import main as make_variable_ini
from c_ini_to_python_dict import main as ini_to_dict
import subprocess
import numpy as np

# --- CONFIGURATION ---
base_dir = Path("~/").expanduser()
chain_dir = base_dir / "chains_mp/planck_TTTEEElensing_mAxi_shooting2025-11-01"
tools_dir = base_dir / "tools"
output_dir = tools_dir / "output"

chain_file = chain_dir / "2025-11-01_10000__8.txt"
log_param_file = chain_dir / "log.param"
output_ini_fixed = output_dir / "output_fixed.ini"
#output_ini_final = tools_dir / "output.ini"

# How many cosmologies to process
N_SAMPLES = 100

npz_file = "all_backgrounds.npz"
    
# Overwrite old file if it exists
if os.path.exists(npz_file):
    os.remove(npz_file)
    print(f"🗑 Cleared old file: {npz_file}")

def main():
    print("=== STEP 1: Create fixed-parameter .ini ===")
    make_fixed_ini(log_param_file, output_ini_fixed)
    weights_array = []

    for i in range(N_SAMPLES):
        output_ini_final = output_dir / f"output_sample_{i}.ini"
        print(f"\n=== STEP 2: Generate .ini for sample {i} ===")
        weights = make_variable_ini(chain_file, log_param_file, tools_dir, output_ini_fixed,output_ini_final)
        weights_array.append(weights)

        print(f"=== STEP 3: Convert .ini to Python dict (JSON) ===")
        ini_to_dict(output_ini_final)

        print(f"=== STEP 4: Run CLASS and append results ===")
        subprocess.run(["python3", "d_run_class.py"], check=True)

    #save weights...
    np.savez("all_weights.npz", weights=np.array(weights_array, dtype=object))

    print("\n=== STEP 5: Done! Background data stored in all_backgrounds.npz ===")
    print("Next step: run e_plot.py to visualize results.")
    subprocess.run(["python", "e_plot.py"], check=True)
    print("✅ Plotting completed.")


if __name__ == "__main__":
    main()

