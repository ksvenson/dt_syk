#!/bin/bash
# Collect SYK and MFIM job timings, including Slurm failures/timeouts.
# ./collect_timings.sh
# ./collect_timings.sh --manifest mfim_manifest.csv
set -o pipefail
MANIFESTS=(sweep_manifest.csv mfim_manifest.csv)
if [[ ${1:-} == --manifest && $# -eq 2 ]]; then
    MANIFESTS=("$2")
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--manifest manifest.csv]" >&2; exit 2
fi

# Display each timing schema separately: MFIM has L/model, SYK has N/g.
# Keep the per-job files separate while jobs run; concurrent appends on Lustre
# would risk losing records. The collector only reads completed files.
shopt -s nullglob
FILES=(timings/*.csv)
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "no files in timings/ yet" >&2
else
    declare -A SCHEMAS=()
    for file in "${FILES[@]}"; do
        IFS= read -r header < "$file"
        [[ -n $header ]] && SCHEMAS["$header"]=1
    done
    for schema in "${!SCHEMAS[@]}"; do
        matching=()
        for file in "${FILES[@]}"; do
            IFS= read -r header < "$file"
            [[ $header == "$schema" ]] && matching+=("$file")
        done
        echo "### completed runs (self-reported; $schema)"
        if command -v column > /dev/null; then
            awk 'FNR==1 && NR>1 {next} 1' "${matching[@]}" | column -s, -t
        else
            awk 'FNR==1 && NR>1 {next} 1' "${matching[@]}"
        fi
        echo
    done
fi

for MANIFEST in "${MANIFESTS[@]}"; do
    [[ -f $MANIFEST ]] || continue
    # Both manifests deliberately keep tag/job_id/submitted in columns 3/4/8.
    ids=$(awk -F, 'NR>1 && $4 ~ /^[0-9]+$/ {print $4}' "$MANIFEST" | paste -sd,)
    [[ -n $ids ]] || continue
    echo "### slurm accounting ($MANIFEST; includes failed and timed-out jobs)"
    if command -v sacct > /dev/null; then
        sacct -X -j "$ids" --format=JobID%10,JobName%20,AllocNodes%5,NCPUS%6,Elapsed%11,Timelimit%11,MaxRSS%10,State%20,ExitCode%8
    else
        echo "sacct unavailable here; job ids: $ids" >&2
    fi
    echo "### submitted but with no timing file"
    for id in ${ids//,/ }; do
        compgen -G "timings/*_${id}.csv" > /dev/null || \
            awk -F, -v i="$id" 'NR>1 && $4==i {printf "  %-14s job %s  (submitted %s)\n", $3, $4, $8}' "$MANIFEST"
    done
    echo
done
