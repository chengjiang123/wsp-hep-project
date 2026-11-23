#!/bin/bash
# Wait for GPU job J (step R) to leave the queue, then collect round R's prompts and split them into batches.
#   usage: [TAG=_llm-sonnet55sub SUB=sonnet] bash collect.sh J R
TAG=${TAG:-_llm-opus55sub}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/${SUB:-.}"
J=$1; R=$2
while ssh -o BatchMode=yes -o ConnectTimeout=20 perlmutter "squeue -j $J -h | grep -q ." 2>/dev/null; do sleep 45; done
ssh -o BatchMode=yes perlmutter "sacct -j $J -X -n -o State | sort | uniq -c; cd \$SCRATCH/wsp_bench/code; module load pytorch/2.13.0 >/dev/null 2>&1; source ../env/bin/activate;
  python handoff_collect.py --tag $TAG --round $R \$SCRATCH/wsp_bench/round${TAG}_$R.json" 2>&1 | grep -v -E "^\*|NOTICE|^ *$" | tail -3
scp -q ${WSP_HOST:-perlmutter}:${WSP_ROOT:?set WSP_ROOT}/round${TAG}_$R.json round$R.json 2>/dev/null
${PY:-python} "$HERE/split.py" round$R.json $R
