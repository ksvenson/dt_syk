#!/bin/bash
# Submit one MFIM job per spin count, with no coupling sweep.
# ./submit_mfim_sweep.sh --dry-run --L 8 10 12
# ./submit_mfim_sweep.sh --L 10 12 --tasks 32 -- --tmax 1000 --n-scr 100
# Resources are conservative requests, not measured runtime predictions.
set -o pipefail
ALLOC=PHY26019
QUEUE=normal
JOBSCRIPT=mfim_job.slurm
MANIFEST=mfim_manifest.csv
L_VALUES=(8 9 10 11 12)
DRY=0
NODES_OVERRIDE=""
TASKS_OVERRIDE=""
TIME_OVERRIDE=""
ANALYSIS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --L) shift; L_VALUES=(); while [[ $# -gt 0 && $1 != --* ]]; do L_VALUES+=("$1"); shift; done ;;
        --tasks) TASKS_OVERRIDE=${2:?missing task count}; shift 2 ;;
        --nodes) NODES_OVERRIDE=${2:?missing node count}; shift 2 ;;
        --time) TIME_OVERRIDE=${2:?missing walltime}; shift 2 ;;
        --account) ALLOC=${2:?missing account}; shift 2 ;;
        --partition) QUEUE=${2:?missing partition}; shift 2 ;;
        --) shift; ANALYSIS_ARGS=("$@"); break ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ ${#L_VALUES[@]} -gt 0 ]] || { echo "provide at least one L" >&2; exit 2; }
[[ -f $JOBSCRIPT && -f mfim_temp_proj.py ]] || { echo "run from the repository directory" >&2; exit 1; }
if [[ $DRY -eq 0 && ! -f .venv/bin/activate ]]; then
    echo "missing .venv/bin/activate" >&2; exit 1
fi
for value in "$NODES_OVERRIDE" "$TASKS_OVERRIDE"; do
    [[ -z $value || $value =~ ^[1-9][0-9]*$ ]] || { echo "nodes/tasks must be positive integers" >&2; exit 2; }
done
mkdir -p tacc_output timings
if [[ $DRY -eq 0 && ! -f $MANIFEST ]]; then
    printf 'L,model,tag,job_id,nodes,tasks,walltime,submitted\n' > "$MANIFEST"
fi
failed=0
for L in "${L_VALUES[@]}"; do
    [[ $L =~ ^[0-9]+$ && $L -ge 3 ]] || { echo "invalid spin count $L" >&2; failed=1; continue; }
    case "$L" in
        3|4|5|6|7|8|9|10) NODES=1; TASKS=32; WALLTIME=04:00:00 ;;
        11|12) NODES=1; TASKS=32; WALLTIME=12:00:00 ;;
        13|14) NODES=2; TASKS=16; WALLTIME=24:00:00 ;;
        *)
            if [[ -z $NODES_OVERRIDE || -z $TASKS_OVERRIDE || -z $TIME_OVERRIDE ]]; then
                echo "L=$L: supply explicit --nodes --tasks --time after estimating dense-matrix memory" >&2
                failed=1; continue
            fi
            NODES=$NODES_OVERRIDE; TASKS=$TASKS_OVERRIDE; WALLTIME=$TIME_OVERRIDE ;;
    esac
    NODES=${NODES_OVERRIDE:-$NODES}; TASKS=${TASKS_OVERRIDE:-$TASKS}; WALLTIME=${TIME_OVERRIDE:-$WALLTIME}
    TAG=MFIM_L${L}
    args=(-A "$ALLOC" -p "$QUEUE" -J "mfim_L${L}" -N "$NODES" -n "$TASKS" -t "$WALLTIME"
          -o "tacc_output/mfim_L${L}_%j.out" -e "tacc_output/mfim_L${L}_%j.err"
          --export="ALL,SWEEP_TAG=$TAG")
    if [[ $DRY -eq 1 ]]; then
        printf '%q ' sbatch "${args[@]}" "$JOBSCRIPT" "$L" "${ANALYSIS_ARGS[@]}"
        printf '\n'
        continue
    fi
    output=$(sbatch "${args[@]}" "$JOBSCRIPT" "$L" "${ANALYSIS_ARGS[@]}" 2>&1)
    status=$?
    # TACC's submission wrapper may print a banner before the numeric job id.
    id=$(printf '%s\n' "$output" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)
    [[ -n $id ]] || id=$(printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)*$/\1/p' | tail -n1)
    if [[ $status -ne 0 || -z $id ]]; then
        printf '%s: submission failed\n%s\n' "$TAG" "$output" >&2; failed=1; continue
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$L" MFIM "$TAG" "$id" "$NODES" "$TASKS" "$WALLTIME" "$(date -Is)" >> "$MANIFEST"
    echo "$TAG: submitted job $id ($TASKS ranks, $NODES nodes)"
done
exit "$failed"
