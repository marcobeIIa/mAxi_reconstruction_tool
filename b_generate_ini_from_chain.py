#!/usr/bin/env python3
import os
import re
import numpy as np
import random
from collections import defaultdict

# --- CONFIG ---
#directory = os.path.expanduser("~/uni/1_master_thesis/montepython_chains/chains_mp/planck_TTTEEElensing_mAxi_shooting2025-11-01/")
#directory_out = os.path.expanduser("~/uni/1_master_thesis/tools")
#chain_file = os.path.join(directory, "2025-11-01_10000__8.txt")
#log_param_file = os.path.join(directory, "log.param")

def read_param_names(log_param_file):
    # === 1. Read parameter names ===

    param_names = []
    scales = {}

    with open(log_param_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue  # skip empty or commented lines

            # Match lines like: data.parameters['omega_b'] = [ 2.2377, None, None, 0.015, 0.01, 'cosmo']
            m = re.match(r"data\.parameters\['(.+?)'\]\s*=\s*\[(.+)\]", stripped)
            if not m:
                continue

            name = m.group(1)
            entries = [e.strip() for e in m.group(2).split(",")]

            # The scale is the second-to-last entry (before the last tag like 'cosmo')
            try:
                scale_str = entries[-2]
                scale = float(scale_str)
            except (ValueError, IndexError):
                scale = 1.0  # default to no scaling if missing or not numeric

            param_names.append(name)
            scales[name] = scale

    print(f"✅ Loaded {len(param_names)} parameters from {log_param_file}")

    return param_names, scales
# === 2. Read the chain file ===

def chain_reader(chain_file):
    # Skip empty files safely
    if os.path.getsize(chain_file) == 0:
        raise ValueError(f"⚠️ Chain file {chain_file} is empty.")

    # Count valid (non-empty) lines
    valid_line_indices = []
    with open(chain_file) as f:
        for idx, line in enumerate(f):
            if line.strip():
                valid_line_indices.append(idx)

    if not valid_line_indices:
        raise ValueError(f"⚠️ No valid samples found in {chain_file}")

    # Pick a random line number
    chosen_idx = random.choice(valid_line_indices)

    # Now read that line
    with open(chain_file) as f:
        for idx, line in enumerate(f):
            if idx == chosen_idx:
                parts = line.strip().split()
                try:
                    weights = float(parts[1])
                    floats = [float(x) for x in parts[2:]]
                    print(f"✅ Loaded random sample from line {idx} with {len(floats)} parameters")
                    return weights,floats
                except ValueError:
                    raise ValueError(f"⚠️ Malformed line at index {idx}: {line.strip()}")
# === 3. Match parameters to values ===
def dict_writer(chain_file,log_param_file):
    param_names, scales = read_param_names(log_param_file)
    weights,sample_values = chain_reader(chain_file)
    
    if len(param_names) != len(sample_values):
        raise ValueError(f"Mismatch: {len(param_names)} names vs {len(sample_values)} values")

    for i,names in enumerate(param_names):
        sample_values[i] *= scales.get(names, 1.0)

    param_dict = dict(zip(param_names, sample_values))


    # === 4. Convert multi-component params for CLASS ===
    # Combine e.g. fraction_maxion_ac__1 ... __9 into single comma-separated strings
    multi_keys = ["fraction_maxion_ac", "theta_ini_mscf"]
    for key_base in multi_keys:
        grouped = [v for k, v in param_dict.items() if k.startswith(key_base + "__")]
        if grouped:
            param_dict[key_base] = ", ".join(str(x) for x in grouped)

    # Remove the individual ones now
    param_dict = {k: v for k, v in param_dict.items() if not any(k.startswith(b + "__") for b in multi_keys)}

    return param_dict,weights

def write_ini(param_dict,directory_out,output_ini_fixed,output_ini_final):
    # === 6. Write temporary CLASS .ini file ===
    output_ini_temp = os.path.join(directory_out, "output_temp.ini")

    with open(output_ini_temp, "w") as f:
        f.write("# CLASS input file auto-generated from MontePython chain\n")
        for k, v in param_dict.items():
            f.write(f"{k} = {v}\n")

    print(f"✅ Wrote temporary CLASS .ini file to {output_ini_temp}")

    # === 7. Merge with fixed parameters file ===
#    output_ini_fixed = os.path.join(directory_out, "output_fixed.ini")
    #output_ini_final = os.path.join(directory_out, "output.ini")

    with open(output_ini_final, "w") as outfile:
        # First the fixed (constant) parameters
        if os.path.exists(output_ini_fixed):
            with open(output_ini_fixed, "r") as fixed:
                outfile.write(fixed.read().strip() + "\n\n")

        # Then the dynamically generated parameters
        with open(output_ini_temp, "r") as dynamic:
            outfile.write(dynamic.read().strip() + "\n")

    print(f"✅ Merged into final CLASS .ini file: {output_ini_final}")

def main(chain_file,log_param_file,directory_out,output_ini_fixed,output_ini_final):
    param_dict,weights = dict_writer(chain_file,log_param_file)
    write_ini(param_dict,directory_out,output_ini_fixed,output_ini_final)
    return weights
