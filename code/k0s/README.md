# Programmed K0S selection: K0S → π+π− vs background

Same idea as `wsp_hep` and `toptag`, on an ALICE task. A standard K0S topological selection is written by hand into network weights, then fine-tuned with a Gaussian prior centred on those weights. New here: a prior whose width per cut comes from the cut's systematic variation.

## Data (`gen.py`)
* Pythia8 pp at 13.6 TeV, SoftQCD:inelastic, K0S and Λ decays switched on (τ0 < 1 m).
* 5 minimum-bias events overlaid at one vertex, to get high-multiplicity combinatorial background.
* Tracking toy, not a full simulation: charged final-state particles, |η| < 0.8, pT > 0.15 GeV, production radius < 40 cm, helices in B = 0.5 T. Smearing at the production point: σpT/pT = 1% ⊕ 0.5% pT, angles 0.5 mrad ⊕ 2 mrad/p, position 20 µm ⊕ 60 µm/pT scaled by (1 + R/10 cm). No material, so no conversions.
* V0 finder: all opposite-sign pairs, vertex from the transverse circle-circle intersection. Preselection: daughter DCA to PV > 0.02 cm, DCA between daughters < 2 cm, cos PA > 0.9, R > 0.2 cm, 0.42 < mππ < 0.58 GeV.
* 40k overlaid events (200k MB): 27.5k true K0S, 41.3k background (9% true Λ, the rest combinatorial). Natural mix about 1.5 background per signal.
* Split: 10k + 10k test, training drawn balanced from the rest. 5 seeds (3 for the raw MLP).

mππ is **not** an input. It is kept only to plot the peak after selection, which is what a real analysis would fit.

## Features (fixed layer, no parameters)
DCA between daughters, cos PA, DCA to PV of each daughter, decay radius, mL/p, Armenteros qT/|α|, and |m(pπ) − mΛ| (smaller of the two hypotheses).

## Hand-written weights (`models.programmed_units`)
| unit | cut | syst. variation Δt |
|---|---|---|
| 0 | DCA daughters < 1.0 cm | 0.25 cm |
| 1 | cos PA > 0.97 | 0.01 |
| 2, 3 | DCA to PV of each daughter > 0.06 cm | 0.02 cm |
| 4 | radius > 0.5 cm | 0.2 cm |
| 5 | mL/p < 20 cm | 5 cm |
| 6 | qT > 0.2 \|α\| | 0.05 |
| 7 | \|m(pπ) − mΛ\| > 5 MeV (competing Λ rejection) | 2 MeV |
| 8, 9 | spare | |

Output: sum of the 8 units − 7, so all pass gives +1 and one failure gives −1. Thresholds are typical ALICE pp strangeness values. The Δt values are illustrative, of the size notes vary cuts by, and are not taken from a specific note.

**Systematics prior** (`models.syst_sigma`): the bias of written unit j gets σ = kj·Δtj, so each threshold may move by about its systematic variation. All other parameters get σ = 0.3.

## Results (test AUC / background rejection at 90% K0S efficiency)
| N | MLP on raw helices | features, random init | prior σ=0.3 | prior σ=1 | prior σ from syst. |
|---|---|---|---|---|---|
| 0 | | | written, no training: 0.984 / 42 | | |
| 100 | 0.683 / 1.3 | 0.970 / 13 | 0.988 / 63 | 0.989 / 69 | 0.989 / 70 |
| 300 | 0.806 / 2 | 0.990 / 60 | 0.992 / 81 | 0.993 / 115 | 0.993 / 107 |
| 1000 | 0.895 / 3 | 0.995 / 177 | 0.994 / 122 | 0.996 / 198 | 0.996 / 225 |
| 3000 | 0.941 / 6 | 0.997 / 317 | 0.997 / 192 | 0.997 / 329 | 0.997 / 275 |
| 30000 | 0.987 / 30 | 0.998 / 603 | 0.998 / 416 | 0.998 / 669 | 0.998 / 613 |

Plain cuts with the same thresholds: εS = 0.79, rejection 58. BDT on the 8 features at 30k: AUC 0.9994, rejection about 5000, but only 2 of the 10k test background candidates pass, so that number is a rough scale only. Rejections above about 500 rest on roughly 20 test background candidates (±25%). Seed spreads are in `scan_*.json`.

### What it says
* **The physics layer is again the big win.** The MLP on raw helix parameters has to learn vertex finding and is at 0.987 / 30 with 30k candidates, below the untrained written net in rejection.
* **The prior helps most at small N.** At N = 100 the written start gives about 5x the rejection of a random start (70 against 13) and AUC 0.989 against 0.970. The gap closes by N ≈ 3000.
* **A tight uniform prior hurts here at large N.** σ = 0.3 ends at 416 against about 600 for the others, because it keeps the cuts near values that are too loose for this detector (next point). This is the opposite of the diphoton toy, where the written cuts were already right.
* **The systematics prior is the best default.** It matches σ = 1 within the seed spread everywhere, and it beats σ = 0.3 at large N, without a σ scan. Its advantage is that each cut is allowed to move by an amount the analysers chose, in the cut's own units.
* **Neither 10-unit net reaches the BDT.** That is the capacity limit, as in toptag.

## Reading the tuned weights (`readout.json`, syst. σ, N = 30k)
| cut | written | tuned | shift / Δt | output weight |
|---|---|---|---|---|
| DCA daughters | < 1.0 cm | < 0.52 cm | −1.9 | 1.5 |
| cos PA | > 0.97 | > 0.991 | +2.1 | 2.0 |
| DCA pos. to PV | > 0.06 cm | > 0.034 cm | −1.3 | 1.9 |
| DCA neg. to PV | > 0.06 cm | > 0.049 cm | −0.5 | 1.6 |
| radius | > 0.5 cm | > 0.46 cm | −0.2 | 1.0 |
| mL/p | < 20 cm | < 10 cm | −2.0 | 1.5 |
| Armenteros | qT > 0.2\|α\| | qT > 0.195\|α\| | −0.1 | 0.7 |
| Λ rejection | > 5 MeV | > 3.6 MeV | −0.7 | 2.3 |

* All written units keep purity ≥ 0.78 (most ≥ 0.98), so they still read as cuts. The random-init net reaches the same AUC and its units mix cos PA and DCA daughters.
* The cuts that tighten (DCA daughters, cos PA, mL/p) are the ones where this toy's tracking is better than the ALICE resolution the written values were chosen for. Signal DCA between daughters has a 95th percentile of 0.14 cm here. So the readout flags exactly the detector mismatch I put in. On real data and MC the same plot would flag where MC tracking differs from data.
* The Λ rejection and the daughter DCA cuts gain weight. qT/|α| loses weight, since the Λ mass veto already does its job.
* At 90% signal efficiency the tuned net leaves about 10x less background under the peak than the written net (B/S 0.004 against 0.039 in the window, natural mix). See the mass panel.

## Caveats
* Tracking toy, not ALICE simulation. No material, no conversions, no PID, no pile-up beyond the overlay, no TPC-only tracks. The point is the method, not the numbers.
* The Δt values are illustrative. The real test is to take them from an analysis note.
* MC truth labels. On data, labels would come from sideband subtraction in mππ, which is possible because mππ is not an input.

## Rerun
```
python gen.py 20000 11 k0s_a.npz; python gen.py 20000 12 k0s_b.npz     # about 10 min each
python run_scan.py feat; python run_scan.py raw
python readout.py; python plot.py dca_dau
```
Needs `pythia8mc torch scikit-learn matplotlib`.
