#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Import your local modules ---
from a_fixed_parameters_to_ini import main as make_fixed_ini
from b_generate_ini_from_chain import main as make_variable_ini
from c_ini_to_python_dict import main as ini_to_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Run mAxi reconstruction tool")
    parser.add_argument("--c", type=str, required=True,
                        help="Directory where the chain files are located")
    parser.add_argument("--o", type=str, required=True,
                        help="Directory where output .ini files will be saved")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of samples to process (default=100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processes (default=4)")
    return parser.parse_args()

args = parse_args()

chain_file = Path(args.c).expanduser()
chain_dir = chain_file.parent
output_dir = Path(args.o).expanduser()
tools_dir = Path(__file__).parent  # main.py location
log_param_file = chain_dir / "log.param"
output_ini_fixed = output_dir / "output_fixed.ini"

os.makedirs(output_dir, exist_ok=True)

npz_file = "all_backgrounds.npz"
if os.path.exists(npz_file):
    os.remove(npz_file)
    print(f"🗑 Cleared old file: {npz_file}")


def process_sample(i):
    """
    Function to process a single sample: generate .ini, convert to dict, run CLASS
    Returns: weight for this sample
    """
    output_ini_final = output_dir / f"output_sample_{i}.ini"
    print(f"[Sample {i}] Generating .ini...")
    weight = make_variable_ini(chain_file, log_param_file, tools_dir, output_ini_fixed, output_ini_final)

    print(f"[Sample {i}] Converting .ini to dict...")
    ini_to_dict(output_ini_final)

    print(f"[Sample {i}] Running CLASS...")
    subprocess.run(["python3", str(tools_dir / "d_run_class.py")], check=True)

    return weight


def main():
    print("=== STEP 1: Create fixed-parameter .ini ===")
    make_fixed_ini(log_param_file, output_ini_fixed)

    N_SAMPLES = args.n
    weights_array = []

    print(f"=== STEP 2-4: Processing {N_SAMPLES} samples in parallel ===")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_sample, i): i for i in range(N_SAMPLES)}

        for future in as_completed(futures):
            i = futures[future]
            try:
                weight = future.result()
                weights_array.append(weight)
                print(f"[Sample {i}] Completed successfully.")
            except Exception as e:
                print(f"[Sample {i}] Failed: {e}")

    # Save all weights
    np.savez("all_weights.npz", weights=np.array(weights_array, dtype=object))

    print("\n=== STEP 5: Done! Background data stored in all_backgrounds.npz ===")
    print("Next step: run e_plot.py to visualize results.")
    subprocess.run(["python3", str(tools_dir / "e_plot.py")], check=True)
    print("✅ Plotting completed.")


if __name__ == "__main__":
    main()

