#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J s3_search
#SBATCH -o /path/to/scratch/wsp_bench/logs/search_%A_%a.log
# Black-box search on the LLM's edit space (search_edits.py). One array task = one (cell, method):
#   BUDGET=256 sbatch -t 02:00:00 --array=1-48 search_edits.sbatch      (16 cells x cmaes, tpe, random)
set -e
cd ${WSP_ROOT:?set WSP_ROOT}/code
module load pytorch/2.13.0
source ../env/bin/activate
export PYTHONUNBUFFERED=1
M=(cmaes tpe random)
i=$((SLURM_ARRAY_TASK_ID - 1))
python search_edits.py --cells cells_search.txt --line $((i / 3 + 1)) --method ${M[$((i % 3))]} --budget ${BUDGET:-256}
