#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -J wsp_handoff
#SBATCH -o /path/to/scratch/wsp_bench/logs/handoff_%A_%a.log
# One step of the file-based edit loop (handoff.py) for one cell per array task.
#   ROUND=1 TAG=_llm-opus55sub CELLS=cells_light.txt sbatch -t 00:30:00 --array=1-16 handoff.sbatch
#   ROUND=5 ...                                           sbatch -t 03:00:00 --array=1-16 handoff.sbatch   (final: pick + FT + test)
set -e
cd ${WSP_ROOT:?set WSP_ROOT}/code
module load pytorch/2.13.0
source ../env/bin/activate
export PYTHONUNBUFFERED=1
python handoff.py step --cells ${CELLS:?} --line ${SLURM_ARRAY_TASK_ID} --tag ${TAG:?} --round ${ROUND:?} --rounds ${ROUNDS:-4} --arms ${ARMS:-ft_wsp,wsp_ft}
