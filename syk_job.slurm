#!/bin/bash
#----------------------------------------------------------------------
# One SYK run at a single (N, g).
#
#   sbatch syk_job.slurm <N> <g>
#
# Normally launched by submit_sweep.sh, which overrides -J/-o/-e/-N/-n/-t
# per system size and exports SWEEP_TAG so the log and timing filenames
# agree. The directives below are only fallbacks for a manual submission.
#----------------------------------------------------------------------
#SBATCH -A PHY26019
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -J syk_sweep
#SBATCH -o tacc_output/syk_%j.out
#SBATCH -e tacc_output/syk_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=ks59683@utexas.edu

set -o pipefail   # deliberately no `set -u`: venv activate scripts trip it

N=${1:?usage: sbatch syk_job.slurm <N_fermions> <g>}
G=${2:?usage: sbatch syk_job.slurm <N_fermions> <g>}

# Filename-safe parameter tag. Normally supplied by the submitter so the
# .out/.err names match; recomputed identically here for manual submissions.
tagify() { printf '%s' "$1" | tr '.-' 'pm'; }
TAG=${SWEEP_TAG:-N${N}_g$(tagify "$G")}

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p computation_cache figs timings

source .venv/bin/activate

# One BLAS thread per rank; otherwise every rank spawns one thread per core.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

echo "=============================================================="
echo " tag        : $TAG"
echo " job id     : $SLURM_JOB_ID    ($SLURM_JOB_NAME)"
echo " N fermions : $N"
echo " g          : $G"
echo " nodes      : $SLURM_NNODES  [$SLURM_JOB_NODELIST]"
echo " mpi tasks  : $SLURM_NTASKS"
echo " started    : $(date -Is)"
echo "=============================================================="

SECONDS=0
ibrun python3 syk_temp_proj.py --N "$N" --g "$G"
rc=$?
elapsed=$SECONDS

printf -v hms '%02d:%02d:%02d' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))

# One file per job: concurrent appends to a shared file are not atomic on Lustre.
{
    printf 'N,g,job_id,job_name,nodes,tasks,seconds,elapsed,exit_code,finished\n'
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$N" "$G" "$SLURM_JOB_ID" "$SLURM_JOB_NAME" "$SLURM_NNODES" \
        "$SLURM_NTASKS" "$elapsed" "$hms" "$rc" "$(date -Is)"
} > "timings/${TAG}_${SLURM_JOB_ID}.csv"

echo "=============================================================="
echo " $TAG  tasks=$SLURM_NTASKS  elapsed=$hms  exit=$rc"
echo "=============================================================="

exit $rc
