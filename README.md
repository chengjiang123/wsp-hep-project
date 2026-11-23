# Open-source material: prompts, LLM interactions and code

Companion material for *Weight-Space Programming with Large Language Models for High Energy and Nuclear Physics*.
Everything an LLM saw and answered in the automated runs is here, together with the code that builds the prompts,
applies the answers to the weights and scores them.

```
prompts/
  encode/   k0s_cut_network.txt, part_pair_interaction.txt         writing weights from physics alone
  edit/     k0s_cut_network_system.txt                              K0S cut-network editor
            pretrained_tagger.txt                                   ParT / OmniLearned editor: system, ops, round template
  extract/  law_reader.txt                                          reading laws from trained weights (3 families)
  subagent_reader_wrapper.txt, subagent_editor_wrapper.txt          instructions given to the Claude Opus 5.5 / Sonnet 5.5 runs
interactions/
  edit_k0s/opus_edit_log.txt                                        an interactive K0S edit loop, round by round
  edit_pretrained/<model>/<task>/<ckpt>/N<N>/s<seed>/<arm>/         every round of every editing loop:
      req<R>.json   the exact system + user prompt of round R
      resp<R>.txt   the model's reply (reasoning + JSON patch)
      after<R>.json validation metrics and diagnostics after applying the patch
  edit_pretrained/results_<model>/...                               final records (chosen round, test AUC before/after FT)
  extract_laws/claude_subagents/                                    batches as given to the readers (neutral codes), answers,
                                                                    code tables and remap.py
  extract_laws/scored/<method>/                                     every scored reading / formula (gpt-oss, Opus, Sonnet,
                                                                    the network itself, PySR, Operon)
code/
  benchmark/   editing pretrained taggers (common.py, llm_edit.py, handoff.py, run_cell.py, compare_light.py, ...)
  laws/        law generation, law-shaped networks, reader, scoring, SR baselines, SRBench reruns
```

## How it works

**Weights with a physical meaning.** Every network the LLM touches is built from, or decomposed into, units whose
weights read as physics: soft cuts, smooth responses, quadratic forms, or the rows of a pretrained output layer.
`code/laws/show.py` prints such a network exactly as the reader sees it; `code/benchmark/common.py:apply_patch` writes an
editor's JSON patch into the weights of ParT or OmniLearned. No gradient is computed in either step.

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
`llm_edit.py:edit_loop` runs the same loop in one process when the LLM is called through an API (gpt-oss-120b via an
OpenAI-compatible gateway, or Claude via `claude_llm.py` and an Anthropic API key).

**Claude Opus 5.5 / Sonnet 5.5 runs.** These answered the identical prompts as subagents: each subagent started from a
fresh context, could open only its batch file (no other files, code, tools or web), treated every item independently,
and wrote one JSON file of answers. Law items carried neutral codes (`item001`, `q001`, `r001`); `remap.py` maps them
back with `neutral*/codes.json`.

## Reproducing

* Laws: `python code/laws/build_reader_prompts.py out.json real b101 b102 b103 b104 b105`, answer the prompts with any
  LLM, then `python code/laws/import_answers.py --tag <name> --model <model> answers.json`.
* Pretrained taggers: see the header of `code/benchmark/handoff.py` (rounds 1 to 5) or `run_cell.py --llm <model>`.
* Paths in the scripts point to the original cluster layout (`$WSP_ROOT`, `$S3_ROOT`); set those variables to your own.
