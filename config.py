import os

BASE_DIR = os.path.expanduser("~/uni/1_master_thesis")
CHAIN_DIR = os.path.join(BASE_DIR, "montepython_chains/chains_mp/planck_TTTEEElensing_mAxi_shooting2025-11-01")
TOOLS_DIR = os.path.join(BASE_DIR, "tools")

CHAIN_FILE = os.path.join(CHAIN_DIR, "2025-11-01_10000__8.txt")
LOG_PARAM_FILE = os.path.join(CHAIN_DIR, "log.param")
FIXED_INI_FILE = os.path.join(TOOLS_DIR, "fixed.ini")
OUTPUT_INI_FILE = os.path.join(TOOLS_DIR, "output.ini")

OUTPUT_DIR = os.path.join(TOOLS_DIR, "output_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STORE_FILE = os.path.join(OUTPUT_DIR, "all_backgrounds.npz")

N_SAMPLES = 10

