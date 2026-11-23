# Three directions on the K0S testbed

\* = LLM-written: written weights, or a written starting point for training.

All three studies use one Pythia8 run (`truth.py`, 320k minimum-bias pp events at 13.6 TeV, true tracks stored).
`reco.py` then rebuilds V0 candidates with adjustable knobs: detector resolution, event overlay, an injected
long-lived particle. The nominal sample (5 overlaid events) has 44k K0S and 66k background candidates.

The method is the toptag winner, soft cuts + one smooth unit per feature (`models_v2.py`), on K0S.

## 1. Tagger improvement (`improve.py`, `improve.png`)

Test set 10k K0S + 25k background, 5 seeds. AUC / rejection at 90% / rejection at 95% K0S efficiency.

| model | N=100 | N=1000 | N=30000 |
|---|---|---|---|
| random start, linear layer | 0.956 / 9 / 7 | 0.992 / 69 / 21 | 0.9975 / 271 / 161 |
| written\*, linear layer, prior max(N,N0) + syst | 0.989 / 69 / 19 | 0.995 / 143 / 67 | 0.9979 / 363 / 179 |
| random start, log layer | 0.972 / 20 / 12 | 0.997 / 480 / 122 | 0.9987 / 5750 / 1005 |
| written\*, log layer, prior 1/N | 0.995 / 170 / 73 | 0.9975 / 1007 / 208 | 0.9989 / 5250 / 1900 |
| written\*, log layer, layer 1 frozen | 0.990 / 60 / 34 | 0.995 / 122 / 56 | 0.9967 / 373 / 103 |
| written\*, log layer, prior max(N,N0) + syst | **0.9956 / 400 / 95** | 0.9975 / 1015 / 207 | 0.9989 / 5083 / 1876 |
| written\* + 48 free units, same prior | 0.9957 / 403 / 89 | 0.9975 / 1067 / 210 | 0.9989 / 5583 / 2127 |
| boosted trees, from scratch | 0.981 / 81 / 20 | 0.997 / 510 / 160 | 0.9992 / 4889 / 1456 |
| boosted trees on top of written net\* | 0.995 / **431** / 73 | **0.9977 / 1132 / 229** | **0.9993 / 6250 / 2256** |

Written net with no training: 0.985 / 41 / 20. Rejections above about 5000 rest on 5 or fewer test background candidates.

* **The physics layer was the bottleneck, not the prior.** DCAs, radius and 1 − cos PA span decades. On a linear scale a tanh unit saturates and the net cannot use the tails. Taking log10 of these features raises rejection at 30k by about 20x for the random start (271 to 5750) and by about 14x for the written net. Cuts stay cuts, because log is monotone, and they still read back in cm.
* **The 1/max(N, N0) prior works.** I recommended it after toptag but had not tested it. At N = 100 it gives 400 against 170 for the plain 1/N prior, and it equals 1/N from N0 = 1000 upward.
* **The written start is worth about 10x the labels at small N.** At N = 100 it reaches 400 at 90% efficiency, against 20 for a random start and 81 for boosted trees.
* **Best overall: boosted trees on top of the written net\*.** The trees start from the tuned written net's log-odds and fit what it misses. They are best or close to best at every N. At 30k they reach 2256 at 95% efficiency against 1456 for trees from scratch. At N = 100 the written net alone is still better at 95% efficiency (95 against 73).
* **Syst. widths and 48 free units add little here.** Freezing layer 1 is the worst written option on K0S, unlike toptag. The soft cuts written by hand are too loose for this toy, so the first layer has to move.

## 2. Interpretability: sensitivity map (`fingerprint.py`, `fingerprint.png`)

Six effects, each switched on alone. Four are regenerated through `reco.py`. Λ yield and K0S pT are done by reweighting the nominal sample.

| effect | how |
|---|---|
| impact-parameter resolution x1.5 | position smearing x1.5 |
| momentum + angle resolution x2 | pT and angle smearing x2 |
| worse resolution for displaced tracks | (1 + R/10 cm) → (1 + R/4 cm) |
| multiplicity x2 | 10 instead of 5 overlaid events |
| Λ yield x2 | true Λ background weighted x2 |
| harder K0S pT spectrum | K0S weighted by exp(pT / 1 GeV) |

For each one the written net\* (log layer, soft + smooth, fixed prior N0 = 1000 so the cuts keep their meaning) is tuned on 30k candidates. The readout has 24 numbers: 8 threshold shifts in units of the syst. variation, 8 cut output weights and 8 smooth-unit weights. Its change from nominal is the effect's fingerprint. Noise is the spread over 6 nominal seeds.

What each fingerprint says, all with |change| > 5 × noise:
* **Impact-parameter resolution:** both daughter DCA-to-PV cuts tighten (+0.09 and +0.12 Δt) and gain weight, and the smooth DCA units go up. Primaries smear out to larger DCA, so the DCA cuts have to work harder.
* **Momentum resolution:** the Λ mass veto widens (+0.13 Δt). Worse mass resolution needs a wider veto window. Almost nothing else moves.
* **Multiplicity:** the Λ-veto weight drops and the smooth DCA units gain. More combinatorics, so the topology matters more and the Λ veto relatively less.
* **Λ yield x2:** only the Λ-veto weight rises (+0.06). This is the cleanest one.
* **Harder K0S pT:** cos PA and the Armenteros cut tighten (+0.18 and +0.16 Δt) and the radius cut shifts out. Faster K0S point better and fly further.
* **Displaced-track resolution:** weak fingerprint. Most K0S here decay within a few cm, where the change is small.

The shifts are small in absolute terms (0.1 to 0.2 of a syst. variation), because the fixed prior holds the cuts. They are still 5 to 10x the seed noise.

**Inverse test.** A mystery sample with impact-parameter resolution x1.3 and multiplicity x1.6, each 0.6 of its single-effect size, is decoded by non-negative least squares on the six fingerprints:

| effect | decoded | injected |
|---|---|---|
| impact-parameter resolution | 0.63 ± 0.04 | 0.6 |
| multiplicity | 0.66 ± 0.07 | 0.6 |
| momentum resolution | 0.05 ± 0.07 | 0 |
| displaced-track resolution | 0.00 ± 0.06 | 0 |
| Λ yield | 0.00 ± 0.01 | 0 |
| K0S pT | 0.16 ± 0.06 | 0 |

Both injected effects come back at the right size. One false lead appears at 2.7σ (K0S pT). For data against MC, this would turn "the tuned net moved" into "which effect, and how much".

## 3. Anomaly detection (`cwola.py`, `anomaly.png`)

I don't know yet what the other project is, so this is the standard LHC set-up: a weakly supervised bump hunt (CWoLa) with no signal labels.

* **Background:** combinatorial V0 candidates with 0.62 < mππ < 0.78 GeV, with no K0S or Λ peak there. 7.7k in the signal region (SR, |m − 0.70| < 0.02 GeV), 23k in the sidebands.
* **Anomaly:** an injected S → π+π− at 0.70 GeV with the K0S momentum spectrum. Two lifetimes: cτ = 3 cm (matches the displaced-vertex prior) and cτ = 0.3 cm (mismatch). 10 to 400 candidates injected in the SR.
* **CWoLa:** classifier on SR vs sidebands, 2-fold, on the 6 mass-independent topological features (log layer). Mass, Armenteros and the Λ veto are left out, since they depend on the mass.
* **Metric:** Z = εS nS / sqrt(εB B_SR) at a true SR background efficiency of 1% (and 0.1% in the json), mean of 3 seeds.

| model | cτ = 3 cm, nS = 20 / 50 / 100 | cτ = 0.3 cm, nS = 20 / 50 / 100 |
|---|---|---|
| no selection | 0.2 / 0.6 / 1.1 | 0.2 / 0.6 / 1.1 |
| written cuts, no training\* | 2.0 / 4.7 / 9.0 | 0.8 / 1.9 / 3.6 |
| CWoLa, random init | 2.2 / 5.3 / 10.8 | 1.8 / 5.1 / 10.6 |
| CWoLa, boosted trees | 0.6 / 3.3 / 9.3 | 0.6 / 2.4 / 7.9 |
| CWoLa, written + prior max(N,N0)\* | 1.4 / 3.6 / 7.7 | 0.9 / 2.6 / 5.9 |
| CWoLa, written + fixed prior\* | 0 / 0 / 0 | 0 / 0 / 0.1 |
| CWoLa, written, layer 1 frozen\* | 1.9 / 4.3 / 8.0 | 0.3 / 0.9 / 1.7 |
| supervised, truth labels (upper bound) | 2.3 / 5.6 / 11.2 | 2.3 / 5.6 / 11.2 |

* **No mass sculpting.** With no signal, every score separates background SR from SB at AUC 0.497 to 0.505.
* **The written cuts alone are a strong model-agnostic selection when the prior fits.** With no training they reach 80 to 87% of the supervised bound for cτ = 3 cm. For cτ = 0.3 cm they reach only about a third, so the prior is a hypothesis, not a free lunch.
* **The written prior does not help CWoLa here, and training on it can hurt.** At low nS the CWoLa labels are almost pure noise. Training pulls the written net off its cuts: the signal-vs-background AUC in the SR drops from 0.993 untrained to about 0.83 with the fixed prior. A random start does better, near the bound, because the injected particles sit where this toy has almost no background.
* **Why the toy is too easy.** Its background has no material interactions, no fake or split tracks and no secondary vertices from heavy flavour. So the displaced region is nearly empty, and any classifier can isolate the signal. In real data that region is full, and that is where a physics prior should matter.

What I would do next for AD: use the written cuts as a fixed preselection or as an extra input to CWoLa, not as the starting weights, and test on ALICE MC with a realistic displaced background.

## Files
`truth.py` (Pythia8, true tracks), `reco.py` (knobs, `run_reco.sh` makes all samples), `models_v2.py`,
`improve.py {nets,trees}`, `fingerprint.py`, `cwola.py {displaced,short}`, `plot_v2.py`.
Results: `improve_*.json`, `fingerprint.json`, `cwola_*.json`, three PNGs. Truth and reco arrays are not included
(100 MB). They regenerate in about 25 min on 2 cores.
