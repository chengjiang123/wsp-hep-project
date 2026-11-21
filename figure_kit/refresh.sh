#!/bin/bash
# Pull the latest results from Perlmutter (small JSONs only), rebuild every CSV, redraw every figure.
#   bash refresh.sh            previews (each figure stamped with how many runs it is based on)
#   bash refresh.sh --final    no stamp
# Needs `ssh perlmutter` working (24 h NERSC key) and ~/opt/anaconda3/bin/python (matplotlib, pandas).
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/opt/anaconda3/bin/python}
R=/path/to/scratch/wsp_bench
mkdir -p ../sync
rsync -a --include='*/' --include='*.json' --exclude='*' perlmutter:$R/story3/results/ ../sync/story3_results/
rsync -a --include='*/' --include='*.json' --exclude='*' perlmutter:$R/results/ ../sync/nscan/
for t in opus55sub sonnet55sub; do
  rsync -a --include='*/' --include='ft_wsp.json' --include='wsp_ft.json' --exclude='*' perlmutter:$R/results_llm-$t/ ../sync/nscan_$t/
done
mkdir -p ../sync/srbench
rsync -a --include='*/' --include='*.json' --include='*.json.updated' --include='*.side' --exclude='*' \
  perlmutter:$R/story3/srbench/results/ ../sync/srbench/results/
rsync -a perlmutter:$R/story3/srbench/monomials.txt ../sync/srbench/ 2>/dev/null || true
$PY export.py
FLAG=$([ "$1" = "--final" ] && echo "" || echo "--preview")
(cd fig_laws && $PY plot.py $FLAG)
(cd fig_coverage && $PY plot.py $FLAG)
(cd fig_nscan && $PY plot.py $FLAG && $PY plot.py --controls $FLAG)
(cd fig_srbench && $PY plot.py $FLAG)
(cd fig_editors && $PY plot.py $FLAG)
(cd fig_edit_search && $PY plot.py $FLAG)
for tm in "top part" "top omni" "qg part" "qg omni"; do
  set -- $tm
  if grep -q "^$1,$2," fig_structure/edit_alone.csv; then (cd fig_structure && $PY plot.py $1 $2 $FLAG); fi
done
