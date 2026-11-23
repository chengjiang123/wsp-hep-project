# Figure kit: numbers, scripts and figures for benchmark v2

One folder per figure. Each holds the numbers (`*.csv`, one row per measurement), a self-contained `plot.py` that
reads only those CSVs, and the rendered `*.pdf` (vector, fonts embedded) and `*.png`. Change a figure by editing the
settings block at the top of its `plot.py` (or `style.py` for fonts, colours, sizes) and running `python plot.py`.

```
bash refresh.sh            # pull new results from Perlmutter, rebuild all CSVs, redraw everything (preview stamp)
bash refresh.sh --final    # same, without the stamp
python export.py           # only rebuild the CSVs from ../sync (no download)
```
Python: matplotlib 3.9, pandas (`refresh.sh` uses `$PY`, default `python`). Raw JSONs live in `../sync/`; nothing here is lost by
re-running `export.py`.

## style.py
Helvetica 7-8 pt, 6.75 in wide, thin marks, light grid. Colours are the first three slots of a palette checked for
colour-blind separation (blue, orange, aqua); every series also has its own marker, so the figures read in grayscale.

## fig_laws/  (Story 3: recovering physics laws)
`plot.py` → `fig_laws.pdf`. (a) laws recovered vs N, (b) out-of-range error / threshold vs N, (c) recovered by family.

| data.csv column | meaning |
|---|---|
| method | `reader_gptoss` (LLM reads the trained law-shaped network's weights), `pysr` (paper setup), `operon` (SRBench estimator), `reader_opus` (1 test call) |
| set, law, family | law set (`real`, `b101`…`b105` hidden sets), law id, family `kin` / `cons` / `decay` |
| N, seed | labelled examples and data seed (the N rows are the same for every method) |
| recovered | 1 if the formula meets the family threshold **outside** the training range (one rule for every method) |
| exact | reader only: exact symbolic reading (the paper's stricter rule) |
| err_in_range, err_out_of_range, threshold | kin: normalised RMS (thr 0.02); cons: 1 − AUC (thr 0.001); decay: mean abs(P − P_true) (thr 0.01) |

`summary.csv`: recovered fraction per method and N (mean and sd over seeds).

## fig_coverage/  (Story 3: where reading the weights works)
`plot.py` → `fig_coverage.pdf`. Reads `../fig_laws/data.csv`. One column per law family, x = N (10², 10³, 10⁴, seed 0:
the 195 instances every reader was run on). Top row: fraction of laws recovered for Opus 5.5 / Sonnet 5.5 / gpt-oss
reading the weights, PySR and Operon, with the trained network itself (grey dashed) for reference. Bottom row: the
same, only where the trained network itself passes the shared rule (`network_pass` = 1). Shaded band: 95–100 %.
`data.csv` in fig_laws now also has the methods `reader_sonnet55sub`, `reader_opus55sub` (Claude subagents, neutral
item codes) and `network`, and a `network_pass` column.

## fig_edit_search/  (paper Figure 4: K0S, LLM weight edits vs black-box search)
`plot.py` -> `fig_edit_search.pdf`. Test rejection R95 of the round chosen on held-back labels vs evaluations, N = 1e4 and
5e4: LLM edits (interactive; automated Opus, 3 seeds), CMA-ES, TPE and random search on the same 24 weights. Trained
models vs label budget as connected curves, one point per N in `REF_NS` (10^2 up to the panel's N, left to right,
log-spaced over the x range of the LLM edit; their x position is not an evaluation count): dashed = BDT / NN from scratch,
solid = WSP + BDT (Opus 5.5 writes the trees, LightGBM boosts on top; ../../../v0_k0s_llmbdt) and WSP + NN (LLM-encoded
network + gradient). Edit `SERIES` / `VSN` / `REF_NS` in `plot.py`; `METRIC` switches to `test_auc`.

| file | columns |
|---|---|
| data.csv | method (`llm_interactive`, `llm_automated`, `cmaes`, `tpe`, `random`), space (`r0` 24 weights of the encoded cuts, `r7` 41 weights of the LLM's structure, `llm`), N, seed, evaluation, test_auc, test_r95 (best round so far by held-back AUC) |
| refs.csv | reference (encoded + gradient, random start + gradient, boosted trees, boosted trees on encoded), N, test_auc, test_r95 |
| refs_vs_N.csv | reference (boosted trees, neural network from random weights, encoded network + gradient, LLM-written trees + boosting), N = 100 to 5e4, seed, test_auc, test_r95 |

Built by `export.py` from the raw runs in `$WSP_DATA` (default `../data`) (v0_k0s/big and cborg/llm_runs).

## fig_editors/  (Story 1 by editing LLM: gpt-oss-120b, Sonnet 5.5, Opus 5.5)
`plot.py` -> `fig_editors.pdf`. Relative gain in background rejection over plain FT (R30 top, R50 quark-gluon) for
FT->WSP and WSP->FT at N = 10^3 (3 seeds), 10^5 and all, with the fixed-rule edit; y range in the settings block,
values beyond it drawn as labelled edge triangles. `data.csv`: test_auc/r30/r50 of each arm with the cell's FT and
fixed-rule values.

## fig_srbench/  (Story 3: SRBench ground-truth laws, against the published table; no LLM)
`plot.py` → `fig_srbench.pdf`. (a) all 130 laws: solution rate vs target noise, our reruns of Operon (published 2021
settings) and PySR over the 14 published methods; (b) the 35 monomial laws (y = C ∏ xᵢ^pᵢ): the LLM reading a fitted
multiplicative unit, the same unit read by a fixed rounding rule (no LLM), and the SR methods; (c) same 35 laws with
exponents **and** constant exact (SRBench's rule accepts any constant factor or offset, so (b) cannot separate them).

| file | columns |
|---|---|
| data.csv | method (`pysr`, `operon2021`, `reader_gptoss`, `rounding`), dataset, seed, noise, monomial (1 = in the 35), r2_test, solved (SRBench's rule), exact (monomial methods only) |
| summary.csv | source (`ours` / `published`), method, subset (`all` / `monomial`), noise, seeds, runs, solved_fraction (mean over seeds), se_over_seeds |
| published.csv | the published ground-truth table (SRBench `results/ground-truth_results.feather`), solved = its `symbolic_solution` |

Protocol: SRBench v2.0 data handling (no scaling, 10,000 training rows drawn with replacement, 75/25 split, the 10
SRBench seeds, noise 0 / 10⁻³ / 10⁻² / 10⁻¹), SRBench's `evaluate_model.py` and symbolic verifier. Operon on 2 seeds.
The reader sees the fitted unit with neutral input names x1…xk only.

## fig_nscan/  (Story 1: WSP vs fine-tuning over the label budget)
`plot.py` → `fig_nscan.pdf`; `plot.py --controls` → `fig_nscan_controls.pdf` (LP-FT, random patch). y = relative gain
in background rejection over plain FT (R30 top, R50 quark–gluon), zoomed to ±40%, values beyond drawn as edge triangles;
the editor of the main scan is gpt-oss-120b.

| data.csv column | meaning |
|---|---|
| task, model | `top` (TopLandscape) / `qg` (quark–gluon); `part` (ParT) / `omni` (OmniLearned-small) |
| N, seed | label budget (80 % train / 20 % validation inside it; `full` = whole training set), seed |
| arm | `ft` plain fine-tune · `ft_wsp` FT then LLM weight edit · `wsp_ft` LLM weight edit then FT · `lpft` logistic head then FT · `script_ft` fixed rule edit then FT · `random_ft` random patch (same tensors, same size) then FT · `scratch` random weights |
| test_auc, test_r30, test_r50 | on the full official test set; R = 1 / background efficiency at 30 % / 50 % signal efficiency |
| auc_after_edit | for edit arms: test AUC right after the edit (or LR head), before any training |

`summary.csv`: mean AUC per arm and the paired difference to `ft` (mean, SE over seeds).

## fig_structure/  (Story 2: what the edit does to the network)
`plot.py [task model]` → `fig_structure_<task>_<model>.pdf`. (a) the LLM's pair-head factors, (b) weight change per
layer during FT, (c) each edit alone vs plain FT.

| file | columns |
|---|---|
| pair_factors.csv | task, model, N, seed, head (1-8), factor (product of the LLM's `pair_scale` ops in its WSP→FT patch; 1 = untouched, 0 = off) |
| displacement.csv | task, model, N, seed, arm, tensor, layer_group, rel_change = ‖θ_end − θ_start‖ / ‖θ_start‖ during fine-tuning |
| edit_alone.csv | task, model, N, seed, what (`LLM edit`, `script edit`, `LR head`, `random patch`), test_auc before any training |

## Status when this kit was made (2026-11-20)
Laws: complete (975 runs per method). N-scan: top/ParT complete; top/OmniLearned, qg/ParT, qg/OmniLearned running.
SRBench Feynman/Strogatz (the planned Story 3 panel c alternative) still running; its CSV and plot get added here when done.
