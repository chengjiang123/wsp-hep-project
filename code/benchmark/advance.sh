#!/bin/bash
# One hand-off for a model's round R: check answer keys against the round's prompts, upload + ingest, submit GPU step
# R+1, and (if R < 4) start the collector for round R+1 in the background (log: <SUB>/collect<R+1>.log).
#   usage: TAG=_llm-sonnet55sub SUB=sonnet bash advance.sh R
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; R=$1
~/opt/anaconda3/bin/python - "$HERE/${SUB:-.}" "$R" <<'PY'
import json, glob, sys
d, R = sys.argv[1], sys.argv[2]
a = {}
for f in glob.glob(f"{d}/r{R}/answers*.json"): a.update(json.load(open(f)))
want = set(json.load(open(f"{d}/round{R}.json")))
print(len(a), "answers; missing:", sorted(want - set(a)), "; unknown keys:", sorted(set(a) - want))
PY
J=$(bash "$HERE/next_round.sh" $R | tail -1); echo "GPU step $((R + 1)): job $J"
[ $R -lt 4 ] && (nohup bash "$HERE/collect.sh" $J $((R + 1)) > "$HERE/${SUB:-.}/collect$((R + 1)).log" 2>&1 &) && echo "collector started"
exit 0
