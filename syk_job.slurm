#!/bin/bash
#----------------------------------------------------------------------
# One SYK run at a single system size.
#
# Do not submit this directly -- use submit_sweep.sh, which overrides
# -J/-o/-e/-n/-t per system size. The directives below are only the
# fallback defaults for a manual `sbatch syk_job.slurm <N>`.
#----------------------------------------------------------------------
#SBATCH -A PHY26019
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 48:00:00
#SBATCH -J syk_temp_proj
#SBATCH -o tacc_output/syk_%j.out
#SBATCH -e tacc_output/syk_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=ks59683@utexas.edu

set -o pipefail   # deliberately no `set -u`: venv activate scripts trip it

N=${1:?usage: sbatch syk_job.slurm <N_fermions>}

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p tacc_output computation_cache figs timings

source .venv/bin/activate

# One BLAS thread per rank. Without this each of the $SLURM_NTASKS ranks
# spawns one thread per core and the node thrashes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Headless plotting; matplotlib is imported only on rank 0 but pin the
# backend anyway so behaviour does not change under idev with X forwarding.
export MPLBACKEND=Agg

echo "=========================================================="
echo " job id     : $SLURM_JOB_ID"
echo " node(s)    : $SLURM_NNODES"
echo " mpi tasks  : $SLURM_NTASKS"
echo " N fermions : $N"
echo " host       : $(hostname)"
echo " started    : $(date -Is)"
echo "=========================================================="

SECONDS=0
ibrun python3 syk_temp_proj.py --N "$N"
rc=$?
elapsed=$SECONDS

printf -v hms '%02d:%02d:%02d' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))

# One file per job: concurrent appends to a shared file are not atomic on Lustre.
printf 'N,job_id,nodes,tasks,seconds,elapsed,exit_code,finished\n%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$N" "$SLURM_JOB_ID" "$SLURM_NNODES" "$SLURM_NTASKS" \
    "$elapsed" "$hms" "$rc" "$(date -Is)" \
    > "timings/N${N}_${SLURM_JOB_ID}.csv"

echo "=========================================================="
echo " N=$N  tasks=$SLURM_NTASKS  elapsed=$hms  exit=$rc"
echo "=========================================================="

exit $rc
