# WSP + BDT on K0S: Opus 5.5 writes the trees, LightGBM boosts on top (2026-11-23)

Same data, test set (30k K0S + 100k background), label samples and boosting recipe as `v0_k0s/improve.py` (plain BDT);
an empty ensemble reproduces the plain BDT. 7 label budgets (1e2 ... 5e4) x 5 seeds, 3 rounds. Each round, two Claude Opus
5.5 subagents (fresh context, only their batch file, no code/tools/web) wrote/edited up to 12 trees of depth <= 4 per
cell from training-part quantiles and, from round 2, held-back metrics of trees alone and trees + boosting, the
boosting's strongest splits and the surviving training background. Round kept per cell: lowest held-back log-loss of
trees + boosting (fixed before the runs). Test metrics never shown to the LLM.

Files: `prompt_system.txt`, `batches/round<R>_<A|B>.md` (exact prompts), `answers/` (replies), `state/` (every round with
held-back and test metrics), `results.json` (chosen round per cell). `python llmbdt.py prep|ingest R`, `final`.

Median test R95 (5 seeds)
| N | 1e2 | 3e2 | 1e3 | 3e3 | 1e4 | 3e4 | 5e4 |
|---|---|---|---|---|---|---|---|
| WSP + BDT (chosen round) | 14 | 49 | 165 | 319 | 641 | 1124 | 1266 |
| plain BDT | 11 | 32 | 159 | 316 | 704 | 1042 | 1333 |
| written trees alone, round 3 | 14 | 102 | 116 | 96 | 116 | 113 | 110 |
| WSP + NN (encoded net + gradient) | 50 | 139 | 231 | 403 | 847 | 1163 | 1389 |

Reading: written trees + boosting = plain BDT within seed scatter at every N. At N = 300-1000 the written trees alone
beat their boosted version on test, which 60-200 held-back labels cannot see (held-back log-loss of the boosted model
is biased low, it is also the early-stopping set).
