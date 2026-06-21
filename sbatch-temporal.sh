#!/bin/bash

#SBATCH -J myjobSYK        # Job name
#SBATCH -o myjob.o%j       # Name of stdout output file
#SBATCH -e myjob.e%j       # Name of stderr error file
#SBATCH -p vm-small        # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 5               # Total # of mpi tasks (2 cores for parallel)
#SBATCH -t 00:10:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A PHY26019        # Project/Allocation name
#SBATCH --mail-user=ks59683@utexas.edu

# 1. Load module and export safe library paths (Replaces LD_PRELOAD)
module load python3/3.9
# export LD_LIBRARY_PATH=/opt/apps/gcc12_2/mvapich2/2.3.7/lib:$LD_LIBRARY_PATH

# 2. Disable CMA to prevent crashes on the vm-small queue
export MV2_SMP_USE_CMA=0

# 3. Activate the virtual environment
source $WORK/syk_installs/syk_env/bin/activate

# 4. Run the python script safely with ibrun
ibrun python3 syk_temporal_basic.py
