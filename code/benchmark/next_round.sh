#!/bin/bash
# After the subagents answered round R: upload answers, write them as replies, submit GPU step R+1 (R+1 = 5: final
# pick + fine-tune + test). Prints the job id.
#   usage: [TAG=.. SUB=.. CELLS=cells_full.txt ARR=1-4 FINAL_T=05:00:00] bash next_round.sh R
set -e
cd "$(dirname "$0")/${SUB:-.}"
R=$1; N=$((R + 1)); T=$([ $N -le 4 ] && echo 00:30:00 || echo 04:00:00)
ssh -o BatchMode=yes perlmutter "mkdir -p \$SCRATCH/wsp_bench/story1_claude/${SUB:-.}/r$R" 2>/dev/null
scp -q r$R/answers*.json perlmutter:/path/to/scratch/wsp_bench/story1_claude/${SUB:-.}/r$R/ 2>/dev/null
ssh -o BatchMode=yes perlmutter "cd \$SCRATCH/wsp_bench/code; module load pytorch/2.13.0 >/dev/null 2>&1; source ../env/bin/activate;
  python handoff.py ingest --tag ${TAG:-_llm-opus55sub} --round $R \$SCRATCH/wsp_bench/story1_claude/${SUB:-.}/r$R/answers*.json;
  if [ $N -le 4 ]; then ROUND=$N TAG=${TAG:-_llm-opus55sub} CELLS=${CELLS:-cells_light.txt} sbatch --parsable -t $T --array=${ARR:-1-16} handoff.sbatch;
  elif [ -n \"${FINAL_T}\" ]; then ROUND=5 TAG=${TAG:-_llm-opus55sub} CELLS=${CELLS:-cells_light.txt} sbatch --parsable -t ${FINAL_T} --array=${ARR:-1-16} handoff.sbatch;
  else ROUND=5 TAG=${TAG:-_llm-opus55sub} CELLS=cells_light.txt sbatch --parsable -t 01:00:00 --array=1-3,5-7,9-11,13-15 handoff.sbatch;
       ROUND=5 TAG=${TAG:-_llm-opus55sub} CELLS=cells_light.txt sbatch --parsable -t 03:00:00 --array=4,8,12,16 handoff.sbatch; fi" 2>&1 | grep -v -E "^\*|NOTICE|^ *$" | tail -2
