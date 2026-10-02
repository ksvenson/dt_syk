#!/bin/bash
#----------------------------------------------------------------------
# Submit one independent job per (N, g), so each gets its own job id and
# its own wall clock. Run from the login node in the submission directory.
#
#   ./submit_sweep.sh                 submit the whole sweep
#   ./submit_sweep.sh --dry-run       print the sbatch commands, submit nothing
#   ./submit_sweep.sh --chain         serialise the g values within each N
#   ./submit_sweep.sh --N 14 16       restrict to these system sizes
#   ./submit_sweep.sh --g 0 100       restrict to these couplings
#----------------------------------------------------------------------
set -o pipefail

ALLOC=PHY26019
QUEUE=normal
JOBSCRIPT=syk_job.slurm
MANIFEST=sweep_manifest.csv

N_VALUES=(18 20 22 24)
G_VALUES=(0 1 10 100)

# Per-N resources: "<nodes> <tasks> <walltime>".
# Keep <tasks> a divisor of `realizations` in syk_temp_proj.py, or the job
# runs at the speed of whichever rank drew an extra realization.
resources_for_N() {
    case "$1" in
        16) echo "1 128 12:00:00" ;;
        18) echo "1 128 12:00:00" ;;
        20) echo "1 128 12:00:00" ;;
        22) echo "1 128 12:00:00" ;;
        24) echo "1 128 12:00:00" ;;
        *)  return 1 ;;
    esac
}

#----------------------------------------------------------------------
DRY=0; CHAIN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --chain)   CHAIN=1; shift ;;
        --N)  shift; N_VALUES=(); while [[ $# -gt 0 && $1 != --* ]]; do N_VALUES+=("$1"); shift; done ;;
        --g)  shift; G_VALUES=(); while [[ $# -gt 0 && $1 != --* ]]; do G_VALUES+=("$1"); shift; done ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -f $JOBSCRIPT ]]        || { echo "missing $JOBSCRIPT" >&2; exit 1; }
[[ -f syk_temp_proj.py ]]  || { echo "missing syk_temp_proj.py" >&2; exit 1; }
[[ -d .venv ]]             || { echo "missing .venv" >&2; exit 1; }

# Slurm opens the log files before the job script runs, so this must exist now.
mkdir -p tacc_output timings

tagify() { printf '%s' "$1" | tr '.-' 'pm'; }

# TACC wraps sbatch with a submission filter that prints a banner and a run of
# "--> Verifying ...OK" lines to STDOUT ahead of the job id, so --parsable is
# not safe to capture directly: $(sbatch --parsable ...) returns the banner too,
# and feeding that to --dependency gives "Job dependency problem". Parse instead.
extract_jobid() {
    local out=$1 id
    id=$(printf '%s\n' "$out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)
    # fall back to a bare numeric line, as --parsable emits ("12345" or "12345;cluster")
    [[ -z $id ]] && id=$(printf '%s\n' "$out" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)*$/\1/p' | tail -n1)
    printf '%s' "$id"
}

total=$(( ${#N_VALUES[@]} * ${#G_VALUES[@]} ))
echo "sweep: ${#N_VALUES[@]} N x ${#G_VALUES[@]} g = $total jobs"
[[ $CHAIN -eq 1 ]] && echo "       chained within each N (<= ${#N_VALUES[@]} at a time)"
if [[ $DRY -eq 0 && ! -f $MANIFEST ]]; then
    echo "N,g,tag,job_id,nodes,tasks,walltime,submitted" > "$MANIFEST"
fi

nsub=0; nfail=0
for N in "${N_VALUES[@]}"; do
    # Capture first: `if ! read ... <<< "$(f)"` would test read's status, and read
    # returns 0 on an empty here-string, so the failure would slip through.
    if ! spec=$(resources_for_N "$N"); then
        echo "  N=$N: no entry in resources_for_N, skipping" >&2
        nfail=$((nfail+1)); continue
    fi
    read -r NODES NTASKS TLIMIT <<< "$spec"
    if [[ -z $NODES || -z $NTASKS || -z $TLIMIT ]]; then
        echo "  N=$N: malformed resources_for_N entry '$spec', skipping" >&2
        nfail=$((nfail+1)); continue
    fi
    dep=""
    for G in "${G_VALUES[@]}"; do
        TAG="N${N}_g$(tagify "$G")"
        args=(
            -A "$ALLOC" -p "$QUEUE"
            -J "syk_$TAG"
            -o "tacc_output/syk_${TAG}_%j.out"
            -e "tacc_output/syk_${TAG}_%j.err"
            -N "$NODES" -n "$NTASKS" -t "$TLIMIT"
            --export="ALL,SWEEP_TAG=$TAG"
        )
        [[ $CHAIN -eq 1 && -n $dep ]] && args+=(--dependency="afterany:$dep")

        if [[ $DRY -eq 1 ]]; then
            printf 'sbatch %s %s %s %s\n' "${args[*]}" "$JOBSCRIPT" "$N" "$G"
            continue
        fi

        out=$(sbatch "${args[@]}" "$JOBSCRIPT" "$N" "$G" 2>&1)
        st=$?
        jobid=$(extract_jobid "$out")
        if [[ $st -ne 0 || -z $jobid ]]; then
            echo "  $TAG: SUBMISSION FAILED (sbatch exit $st)" >&2
            printf '%s\n' "$out" | sed 's/^/      | /' >&2
            nfail=$((nfail+1)); continue
        fi
        if [[ $CHAIN -eq 1 ]]; then
            dep=$jobid        # numeric only; anything else breaks --dependency
        fi
        nsub=$((nsub+1))
        printf '  %-14s job %-10s %s node(s), %-4s tasks, %s\n' \
            "$TAG" "$jobid" "$NODES" "$NTASKS" "$TLIMIT"
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$N" "$G" "$TAG" "$jobid" "$NODES" "$NTASKS" "$TLIMIT" "$(date -Is)" \
            >> "$MANIFEST"
    done
done

[[ $DRY -eq 1 ]] && exit 0
echo "submitted $nsub, failed $nfail; manifest in $MANIFEST"
cat <<'EOF'

  squeue -u $USER -o "%.12i %.22j %.9P %.11l %.11M %.6D %.2t %R"
  squeue -u $USER --start            # estimated start times
  scontrol show reservation          # maintenance window blocking long jobs?
  ./collect_timings.sh               # once things finish
EOF
