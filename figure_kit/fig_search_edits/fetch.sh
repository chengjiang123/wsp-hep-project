#!/bin/bash
# Pull E1 results from Perlmutter into ../../sync (needs a valid NERSC ssh key), then build data.csv.
set -e
R=/path/to/scratch/wsp_bench
S=$(cd "$(dirname "$0")/../.." && pwd)/sync
mkdir -p "$S"
rsync -a perlmutter:$R/search_edits "$S/"
for t in opus55sub sonnet55sub; do
  rsync -a --include='*/' --include='ft.json' --include='ft_wsp.json' --exclude='*' perlmutter:$R/results_llm-$t/ "$S/results_llm-$t/"
done
python "$(dirname "$0")/make_csv.py"
