# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # MCMC analysis notebook
# ## using the vanilla MontePython analysis tools
# (there si another notebook using getdist, getdist_mcmc.ipynb, which should be preferred. this notebook was used for simplicity, and is now legacy)

# +
import subprocess
from pathlib import Path

bases = ["/Users/bellamarco01/uni/1_master_thesis/montepython_chains/chains_mp/9_axions/", "/Users/bellamarco01/uni/1_master_thesis/montepython_chains/chains_mp/4_axions"]

chains = [
        "planck_TTTEEElensing_pantheon_plus_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_sh0es_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_desi_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_sh0es_desi_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_boss_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_sh0es_boss_2025-12-19",
         ]

chains2 = [
        "planck_TTTEEElensing_2025-12-23",
        "planck_TTTEEElensing_pantheon_plus_2025-12-23",
        "planck_TTTEEElensing_pantheon_plus_sh0es_2025-12-23",
        "planck_TTTEEElensing_pantheon_plus_sh0es_desi_2025-12-23",
         ]

for chain in chains:
    cmd = [
        "python",
        "/Users/bellamarco01/uni/1_master_thesis/montepython/montepython_public/montepython/MontePython.py",
        "info",
        str(Path(bases[0]) / chain)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

for chain in chains2:
    cmd = [
        "python",
        "/Users/bellamarco01/uni/1_master_thesis/montepython/montepython_public/montepython/MontePython.py",
        "info",
        str(Path(bases[1]) / chain)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# +
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

base = Path("/Users/bellamarco01/uni/1_master_thesis/montepython_chains/chains_mp/9_axions")
base2 = Path("/Users/bellamarco01/uni/1_master_thesis/montepython_chains/chains_mp/4_axions")


chains = [
        "planck_TTTEEElensing_pantheon_plus_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_sh0es_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_desi_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_sh0es_desi_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_boss_2025-12-19",
        "planck_TTTEEElensing_pantheon_plus_sh0es_boss_2025-12-19",
         ]

chains2 = [
        "planck_TTTEEElensing_pantheon_plus_2025-12-23",
        "planck_TTTEEElensing_pantheon_plus_sh0es_2025-12-23",
        "planck_TTTEEElensing_pantheon_plus_desi_2025-12-23",
        "planck_TTTEEElensing_pantheon_plus_sh0es_desi_2025-12-23",
         ]

for chain_name in chains:
    chain_dir = base / chain_name / "plots"
    chain_dir2 = base2 / chain_name / "plots"
    pdfs = list(chain_dir.glob("*.pdf"))
    pdfs2 = list(chain_dir2.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {chain_dir}")
        continue
    if not pdfs2:
        print(f"No PDFs found in {chain_dir}")
        continue

    for pdf_path in pdfs:
        print(f"Displaying {pdf_path.name}")
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()  # Render page to pixmap
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)  # Convert to array

            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    for pdf_path in pdfs2:
        print(f"Displaying {pdf_path.name}")
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()  # Render page to pixmap
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)  # Convert to array

            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

# -




