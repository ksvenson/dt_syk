#!/bin/bash
#----------------------------------------------------------------------
# Aggregate the per-job timing files and cross-check against Slurm's own
# accounting, which also records jobs killed by the wall clock (those never
# get to write their own timing file).
#
#   ./collect_timings.sh              > sweep_timings.csv
#----------------------------------------------------------------------
set -o pipefail

MANIFEST=sweep_manifest.csv

if ! compgen -G "timings/*.csv" > /dev/null; then
    echo "no files in timings/ yet" >&2
else
    echo "### completed runs (self-reported)"
    awk 'FNR==1 && NR>1 {next} 1' timings/*.csv | column -s, -t
    echo
fi

if [[ -f $MANIFEST ]]; then
    ids=$(awk -F, 'NR>1 && $4 ~ /^[0-9]+$/ {print $4}' "$MANIFEST" | paste -sd,)
    if [[ -n $ids ]]; then
        echo "### slurm accounting (authoritative; includes failed and timed-out jobs)"
        sacct -X -j "$ids" \
              --format=JobID%10,JobName%20,AllocNodes%5,NCPUS%6,Elapsed%11,Timelimit%11,MaxRSS%10,State%20,ExitCode%8
        echo
        echo "### submitted but with no timing file (still queued, running, or killed)"
        for id in ${ids//,/ }; do
            compgen -G "timings/*_${id}.csv" > /dev/null || \
                awk -F, -v i="$id" 'NR>1 && $4==i {printf "  %-14s job %s  (submitted %s)\n", $3, $4, $8}' "$MANIFEST"
        done
    fi
else
    echo "no $MANIFEST; run submit_sweep.sh first" >&2
fi
