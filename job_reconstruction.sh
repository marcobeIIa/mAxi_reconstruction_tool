#!/bin/bash

# SLURM options:

#SBATCH --job-name=reconstruction    # Nom du job
#SBATCH --output=job-%j.log   # Standard output et error log

#SBATCH --partition=lupm               # Choix de partition
#SBATCH --nodelist=c1

#SBATCH --cpus-per-task=16            #Nombre de COEURS PAR NOEUD (Max 128)
#SBATCH -N 1                          #Nombre de NOEUDS (Max 4)

#SBATCH --ntasks-per-node=1


#SBATCH --mem-per-cpu=4024            # Mémoire en MB par défaut par COEUR (Max 4024)

#SBATCH --mail-user=marco.bella@studenti.unitn.it   # Où envoyer l'e-mail
#SBATCH --mail-type=END,FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)


#SBATCH --time=336:00:00

###################
cd ~/mAxi_reconstruction_tool
source ~/.bashrc
conda init
conda activate mpi_clean

mpirun -np 16 python main.py --o ~/mAxi_reconstruction_tool/output1 --c /home/bella/chains_mp/planck_TTTEEElensing_mAxi_shooting2025-11-01/2025-11-01_10000__8.txt
