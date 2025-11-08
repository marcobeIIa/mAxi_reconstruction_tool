"""
Convert a CLASS .ini file into a Python dictionary.
"""

#ini_file = "output.ini"  # or any other .ini

params_mEDE = {}

def main(ini_file):
    with open(ini_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty or commented lines

            if "=" not in line:
                print(f"⚠️ Skipping malformed line: {line}")
                continue

            key, val_str = map(str.strip, line.split("=", 1))

            # Always treat as string if it contains comma, otherwise try float
            try:
                if "," in val_str:
                    val = val_str  # keep comma-separated as string
                else:
                    val_float = float(val_str)
                    val = val_float
            except ValueError:
                val = val_str  # fallback as string

            params_mEDE[key] = val

    import json

    with open("params_mEDE.json", "w") as f:
        json.dump(params_mEDE, f)

