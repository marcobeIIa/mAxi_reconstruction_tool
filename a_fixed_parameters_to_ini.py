import re
import os

#directory = os.path.expanduser("~/uni/1_master_thesis/montepython_chains/chains_mp/planck_TTTEEElensing_mAxi_shooting2025-11-01/")
#directory_out = os.path.expanduser("~/uni/1_master_thesis/tools")
#log_param_file = os.path.join(directory, "log.param")
#output_ini = os.path.join(directory_out, "output_fixed.ini")

def read_params(log_param_file):
    fixed_params = {}

    with open(log_param_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Match lines like: data.cosmo_arguments['N_ur'] = 2.0328
            m = re.match(r"data\.cosmo_arguments\['(.+)'\]\s*=\s*(.+)", line)
            if m:
                key = m.group(1)
                val_str = m.group(2).strip()

                # Remove quotes if present
                if val_str.startswith("'") and val_str.endswith("'"):
                    val = val_str[1:-1]
                elif val_str.startswith('"') and val_str.endswith('"'):
                    val = val_str[1:-1]
                else:
                    # Try to detect integer first
                    if re.match(r"^-?\d+$", val_str):
                        val = int(val_str)
                    else:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = val_str  # fallback as string

                fixed_params[key] = val
    return fixed_params

def write_fixed_ini(fixed_params,output_ini):
    with open(output_ini, "w") as f:
        f.write("# CLASS ini file: first part, constant parameters (from log.param)\n\n")
        for key, val in fixed_params.items():
            f.write(f"{key} = {val}\n")
    print(f"✅ Wrote .ini file with {len(fixed_params)} parameters to {output_ini}")

def main(log_param_file,output_ini):
    fixed_params = read_params(log_param_file)
    write_fixed_ini(fixed_params,output_ini)


