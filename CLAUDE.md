# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Public release of the code behind *Weight-Space Programming with LLMs for HEP/NP*. Read `README.md` first; it has the
layout and the table of environment variables (`WSP_ROOT`, `S3_ROOT`, `WSP_DATA`, ...) every script relies on.

## Commands
- No build, test suite or linter. `pip install -r requirements.txt`; check edits with `python -m py_compile <file>`.
- Redraw a figure: `cd figure_kit/<fig> && python plot.py`. Rebuild CSVs from raw JSONs: `python figure_kit/export.py`.
- Pretrained-tagger edit loop: `python code/benchmark/handoff.py step|ingest ...` (one round per call, documented in its
  header); single process with an API LLM: `python code/benchmark/run_cell.py --llm <model>`.
- Law study: `code/laws/build_reader_prompts.py` -> any LLM -> `code/laws/import_answers.py`.

## Architecture
- Three operations on physics-shaped weights, each with a non-LLM control: encode / edit (K0S cut networks in
  `code/k0s/`, decision trees in `code/bdt/`, pretrained-tagger heads in `code/benchmark/`) and extract (`code/laws/`).
- `code/benchmark/common.py` is the hub: dataset/checkpoint loading (ParT via weaver, OmniLearned), metrics (R30/R50),
  and `apply_patch`, which maps an LLM's JSON patch onto a 31-number edit space. `handoff.py`, `llm_edit.py`,
  `search_edits.py` and `run_cell.py` all go through it. A "cell" = task x checkpoint x N x seed x arm, listed one per
  line in `cells_*.txt`; `*.sh` helpers drive the loop on a SLURM cluster over ssh (`$WSP_HOST`).
- Round selection always uses held-back validation labels; test metrics are never shown to the LLM. LLM prompts are
  kept verbatim in `prompts/` and must stay in sync with the code that builds them.
- Never commit data, checkpoints, run outputs or API keys; keys come only from the environment.
