#!/usr/bin/env bash
# RenderAll.sh
#
# Remake every table and figure of the paper from the records in Records/: all the
# *Tables.jl and *Figures.jl scripts and the two schematics, then FinalizeFigures.py.
# The Runs scripts are not run. Each script's printed output, including the numbers the
# text quotes, goes to Logs/<script>.txt. A failed script is reported at the end and
# does not stop the others.
#
# Needs Julia and Python with numpy, pypdf and PyMuPDF. The bootstrap intervals depend on
# the Julia version, so the paper's quoted intervals must come from the version used here;
# it is printed at the start and saved in Logs/julia_version.txt.
# Run:     bash RenderAll.sh
#          JULIA="julia +1.13.1" bash RenderAll.sh   (default "julia +release")
#          PYTHON=python3 bash RenderAll.sh          (the default)
# Writes:  Tables/*.tex, Figures/*.pdf and Logs/*.txt

set -u
cd "$(dirname "$0")"

JULIA=${JULIA:-"julia +release"}
PYTHON=${PYTHON:-python3}

version=$($JULIA -e 'print(VERSION)') || { echo "Cannot start Julia with: $JULIA"; exit 1; }
echo "Julia $version"

tables="FunctionalTables ConfirmationTables AdequacyTables KvsTTables StatisticalTables PrefixTables
        BadCoarseTables PredictionTables LongChunkTables"
figures="TimeGridSchematic ShadowingSchematic ErrorContours KvsTFigures StatisticalFigures PrefixFigures
         AdequacyFigures PredictionFigures LongChunkFigures"

mkdir -p Logs
echo "$version" > Logs/julia_version.txt
failed=""
for name in $tables $figures; do
    printf '%-20s ' "$name"
    start=$SECONDS
    if $JULIA "$name.jl" > "Logs/$name.txt" 2>&1; then
        echo "done ($((SECONDS - start)) s)"
    else
        echo "FAILED, see Logs/$name.txt"
        failed="$failed $name"
    fi
done

printf '%-20s ' "FinalizeFigures"
if $PYTHON FinalizeFigures.py > Logs/FinalizeFigures.txt 2>&1; then
    echo "done"
else
    echo "FAILED, see Logs/FinalizeFigures.txt"
    failed="$failed FinalizeFigures"
fi

if [ -n "$failed" ]; then
    echo "Failed:$failed"
    exit 1
fi
echo "All tables and figures remade; printed numbers are in Logs/."
