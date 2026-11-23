# Hand editing in weight space, K0S (`handedit.py`, `handedit.png`)

\* = LLM-written.

**Question:** how far does pure LLM weight editing go, with no gradient steps, before training on data wins?

**Set-up**
* Start from the written log-layer net\* with no training.
* Each round, the LLM (me) reads diagnostics computed only on the same 1000 labelled candidates (500 K0S + 500 background) that the gradient runs used. The diagnostics are per-unit pass rates, background survivors and K0S losers at 95% efficiency, which unit votes for or against them, and feature quantiles.
* I then edit thresholds, sharpness and output weights by hand, or write a new unit. Every round is a JSON spec in `hand/`.
* The test set (10k K0S + 25k background) was used once, after the last round. The final round was chosen in advance by the best edit-set AUC.

| round | edit | test AUC | rej. 90% | rej. 95% |
|---|---|---|---|---|
| r0 | written net\*, no edits | 0.9851 | 41 | 20 |
| r1 | tighten cos PA (0.998), DCA daughters (0.25 cm), mL/p (12 cm); loosen radius and Armenteros | 0.9899 | 49 | 12 |
| r2 | Λ veto becomes a real veto (weight 3); Armenteros back as a Λ handle; cos PA 0.995 | 0.9891 | 269 | 69 |
| r3 | radius < 40 cm (impossible vertices); two-input unit d_V0 = L·θ < 0.1 cm | 0.9919 | 338 | 94 |
| r4 | graded second steps: cos PA > 0.999, DCA daughters < 0.1 cm | 0.9949 | 500 | 122 |
| r5 | second Armenteros step (mixed, not kept) | 0.9936 | 532 | 108 |
| r6 | smooth ramps on daughter DCAs and radius, read from edit-set log(S/B) | 0.9955 | 472 | 145 |
| **r7** | **gentler ramps (final)** | **0.9956** | **532** | **155** |

Gradient training on the same 1000 candidates (mean of 5 seeds):

| model | AUC | rej. 90% | rej. 95% |
|---|---|---|---|
| written\* + gradient (prior max(N,N0) + syst) | 0.9975 | 1015 | 207 |
| boosted trees, from scratch | 0.9968 | 510 | 160 |
| random start + gradient | 0.9969 | 480 | 122 |
| written\*, gradient on output layer only | 0.9945 | 122 | 56 |

## What it says
* **Hand editing works.** Seven rounds took the written net\* from 41 to 532 in rejection at 90% (13x) and from 20 to 155 at 95% (8x), with no gradient step.
* **It matches generic training on the same data.** It is level with boosted trees and a random-start network trained on the same 1000 labels in rejection. It is lower in AUC (0.9956 against 0.997).
* **It beats gradient training of the output layer alone** (4x at 90%), because hand edits can move the cuts themselves.
* **Gradient tuning of the same written net\* is still better:** about 2x the rejection at 90% and AUC 0.9975. Hand editing reaches the level that gradient tuning gets from roughly 100 to 300 labels.
* **Where data wins:** once the edit set runs out of resolution. After round 4, only 1 or 2 of the 500 labelled background candidates survive, so diagnostics can't guide finer edits. Gradient descent uses every candidate's small contribution, which hand reading can't.
* **The edits are physics, and readable.** A veto must outweigh the other votes. Impossible vertices (R > 40 cm) are cut. Pointing is replaced in part by d_V0 = L·θ for short decays. Graded steps act as a coarse likelihood ratio. The test moved in the same direction as the edit set at every round, so there is no sign of overfitting by hand.
* **Two edits made things worse** (r1 and r5). Both were visible on the edit set and corrected or dropped.

## Files
`handedit.py diag hand/rN.json` (diagnostics), `handedit.py test ...` (test, once), `hand/r0..r7.json`, `handedit_test.json`, `plot_hand.py`.
