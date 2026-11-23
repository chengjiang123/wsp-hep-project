#!/bin/bash
# Full-N hand-off for one model:  bash advance_full.sh opus|sonnet R
M=$1; R=$2
TAG=_llm-${M}55sub SUB=${M}_full CELLS=cells_full.txt ARR=1-4 FINAL_T=05:00:00 bash "$(dirname "$0")/advance.sh" $R
