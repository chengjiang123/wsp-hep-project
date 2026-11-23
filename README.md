# Weight-Space Programming with LLMs for High Energy and Nuclear Physics

Code, prompts and LLM interactions for *Weight-Space Programming with Large Language Models for High Energy and
Nuclear Physics* (Cheng Jiang et al., arXiv link to follow). Project page: `docs/` (GitHub Pages).

An LLM operates directly on the weights of physics-shaped networks, built from or decomposed into soft cuts, smooth
responses and quadratic forms, so that every weight has a physical meaning. It **encodes** physics knowledge into the
weights, **edits** them without gradient steps, and **extracts** governing laws from trained weights. Every operation
is paired with a control that uses no LLM.

```
code/k0s/         K0S -> pi+ pi- selection: toy generator, cut networks, LLM edit loop, black-box search baseline
code/bdt/         WSP + BDT: an LLM writes decision trees, LightGBM boosts on top
code/benchmark/   editing pretrained taggers (ParT, OmniLearned) on top / quark-gluon, black-box search on the same edits
code/laws/        law generation, law-shaped networks, LLM reader, scoring, PySR / Operon / SRBench baselines
prompts/          every prompt given to an LLM (encode/, edit/, extract/, subagent wrappers)
results/k0s_bdt/  WSP + BDT run: exact prompts, LLM answers, every round's metrics, chosen round per cell
figure_kit/       one folder per figure: the numbers (CSV) and a self-contained plot.py
```

## Install
```
pip install -r requirements.txt
```
The pretrained-tagger benchmark also needs the [weaver](https://github.com/hqucms/weaver-core) Particle Transformer
code and the OmniLearned package and checkpoints; the SR baselines need PySR (Julia) and pyoperon.

## Where your data lives: environment variables
Datasets, checkpoints and run outputs are not in this repository. The scripts find them through environment variables;
the defaults in the code (`/path/to/scratch/...`) are placeholders, so set the ones you need first:

| variable | used by | what it points to |
|---|---|---|
| `WSP_ROOT` | `code/benchmark/`, `*.sbatch`, `figure_kit/refresh.sh` | work area of the pretrained-tagger benchmark: prepared JetClass-style datasets, `code/` copy, `env/`, `results*/`, `handoff*/`, `search_edits/`, `logs/` |
| `PART_DIR` | `code/benchmark/common.py` | folder with the Particle Transformer checkpoints |
| `OMNI_CKPT` | `code/benchmark/common.py` | OmniLearned pretrained checkpoint (`.pt`) |
| `S3_ROOT` | `code/laws/` | work area of the law study: generated laws, trained law-shaped nets, reader answers, `srbench/` (conventionally `$WSP_ROOT/story3`) |
| `SR_DIR`, `SRBENCH` | `code/laws/sr_laws.py`, `srbench_run.py` | symbolic-regression outputs and an SRBench checkout |
| `WSP_CODE` | `code/laws/reader.py` | folder holding `code/benchmark` on the cluster (for the shared helpers) |
| `WSP_DATA` | `code/bdt/`, `figure_kit/export.py` | local raw data, default `data/` at the repo root (e.g. `v0_k0s/big/nom.npz` from `code/k0s/gen.py`) |
| `BDT_OUT` | `code/bdt/llmbdt.py` | where the WSP + BDT loop reads/writes batches, answers, state; default `results/k0s_bdt` |
| `WSP_HOST` | `code/benchmark/*.sh`, `figure_kit/*.sh` | ssh alias of the cluster that holds `$WSP_ROOT` (default `perlmutter`) |
| `PY` | shell helpers | Python interpreter to use (default `python`) |

LLM access: `LLM_URL` and `LLM_MODEL` (any OpenAI-compatible endpoint), with `CBORG_API_KEY` or `OPENAI_API_KEY`;
`ANTHROPIC_API_KEY` for `code/benchmark/claude_llm.py`. Keys are read from the environment only.

The `*.sbatch` files are written for NERSC Perlmutter (`-q shared -C gpu -G 1`); replace `#SBATCH -A <your_account>`
and the `#SBATCH -o` log path (SLURM does not expand variables there) with your own.

## How it works

**Weights with a physical meaning.** `code/laws/show.py` prints a network exactly as the reader sees it;
`code/benchmark/common.py:apply_patch` writes an editor's JSON patch into the weights of ParT or OmniLearned. No
gradient is computed in either step.

**Extract (laws).**
1. `gen.py` draws hidden laws (kinematic invariants, conservation laws, decay in flight; most of them unphysical),
   `nets.py` trains a law-shaped network on N labels per law.
2. `build_reader_prompts.py` writes the prompt for each network: the family format (`prompts/extract/law_reader.txt`)
   followed by the printed weights. The model never sees data or truth.
3. The reader returns a JSON answer; `import_answers.py` (or `reader.py` for API models) scores it with one rule for all
   methods: the stated formula must reach the precision of the true law **outside** the training range.
   `sr_laws.py` runs PySR and Operon on the same labels; `net_fidelity.py` scores the trained network itself.

**Edit (pretrained taggers).** One loop = 4 rounds on one cell (task, checkpoint, label budget N, seed, arm):
1. GPU step (`handoff.py step`): rebuild the network, replay the patches so far, score the newest one on the validation
   part of the budget, compute diagnostics on the training part, and write the next prompt `req<R>.json`.
2. The LLM answers with a patch (`resp<R>.txt`); `handoff.py ingest` stores it.
3. After round 4, the round with the lowest validation loss is kept; for *WSP -> FT* the edited network is then
   fine-tuned with the published recipe of the checkpoint, and both arms are scored once on the full test set.

`llm_edit.py:edit_loop` runs the same loop in one process when the LLM is called through an API. `search_edits.py`
runs CMA-ES, TPE and random search on the same edit space for the black-box comparison.

**Claude Opus 5.5 / Sonnet 5.5 runs.** These answered the identical prompts as subagents: each started from a fresh
context, could open only its batch file (no other files, code, tools or web), treated every item independently, and
wrote one JSON file of answers. Law items carried neutral codes (`item001`, `q001`, `r001`) that `remap.py` maps back.

## Reproducing
* Laws: `python code/laws/build_reader_prompts.py out.json real b101 b102 b103 b104 b105`, answer the prompts with any
  LLM, then `python code/laws/import_answers.py --tag <name> --model <model> answers.json`.
* Pretrained taggers: see the header of `code/benchmark/handoff.py` (rounds 1 to 5) or `run_cell.py --llm <model>`.
* K0S: `python code/k0s/gen.py n_events seed out.npz`, then `code/k0s/README.md`.
* Figures: `cd figure_kit/<fig> && python plot.py` (reads only the CSVs next to it).

## Citation
```bibtex
@article{Jiang:2026wsphep,
  author = {Jiang, Cheng and others},
  title  = {Weight-Space Programming with Large Language Models for High Energy and Nuclear Physics},
  year   = {2026},
  note   = {arXiv number to follow}
}
```
