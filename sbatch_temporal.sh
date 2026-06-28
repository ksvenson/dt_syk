#!/bin/bash

#SBATCH -A PHY26019                     # Project/Allocation name
#SBATCH -J temporal                     # Job name
#SBATCH -o tacc_output/temporal_%j.out  # Name of stdout output file
#SBATCH -e tacc_output/temporal_%j.err  # Name of stderr error file
#SBATCH -p normal                       # Queue (partition) name
#SBATCH -N 1                            # Total # of nodes 
#SBATCH -n 1                            # Total # of mpi tasks
#SBATCH -t 02:00:00                     # Run time (hh:mm:ss)
#SBATCH --mail-type=all                 # Send email at begin and end of job
#SBATCH --mail-user=ks59683@utexas.edu

ibrun python3 main.py
