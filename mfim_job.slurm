#!/bin/bash
# One periodic MFIM chain per job; MPI ranks split time intervals within it.
# sbatch mfim_job.slurm <L_spins> [analysis options...]
#SBATCH -A PHY26019
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 12:00:00
#SBATCH -J mfim
#SBATCH -o tacc_output/mfim_%j.out
#SBATCH -e tacc_output/mfim_%j.err
set -o pipefail

L=${1:?usage: sbatch mfim_job.slurm <L_spins> [analysis options...]}
TAG=${SWEEP_TAG:-MFIM_L${L}}
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p computation_cache figs timings
source .venv/bin/activate || exit 1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

echo "$TAG: job=$SLURM_JOB_ID nodes=$SLURM_NNODES ranks=$SLURM_NTASKS started=$(date -Is)"
SECONDS=0
ibrun python3 mfim_temp_proj.py --L "$L" "${@:2}"
rc=$?
elapsed=$SECONDS
printf -v hms '%02d:%02d:%02d' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
{
    printf 'L,model,job_id,job_name,nodes,tasks,seconds,elapsed,exit_code,finished\n'
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$L" MFIM "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "$SLURM_NNODES" \
        "$SLURM_NTASKS" "$elapsed" "$hms" "$rc" "$(date -Is)"
} > "timings/${TAG}_${SLURM_JOB_ID}.csv"
echo "$TAG: ranks=$SLURM_NTASKS elapsed=$hms exit=$rc"
exit "$rc"
