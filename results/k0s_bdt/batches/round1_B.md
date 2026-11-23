You are a particle physicist building a K0S -> pi+ pi- selection as a small set of boosted decision trees, by writing
the branches yourself: which feature each node splits on, the threshold in physical units, and the log-odds in each
leaf. No training is done on your trees. After you answer, a standard gradient-boosting library (LightGBM, the same
recipe for every item) adds its own trees on top of yours using the training labels of the item, with early stopping on
the held-back labels; your trees are its starting point. The final selection is your trees plus the boosted trees.

The sample.
  Pythia8 minimum-bias pp collisions at sqrt(s) = 13.6 TeV, each with 5 more minimum-bias events overlaid.
  A toy of an ALICE-like barrel tracker: charged tracks with pT > 0.15 GeV, |eta| < 0.8, produced within 40 cm of the beam.
  Position resolution (20 um + 60 um / pT[GeV]) * (1 + R / 10 cm), pT resolution 1% (+) 0.5% * pT,
  angular resolution 0.5 mrad (+) 2 mrad / p. No material, no fake tracks, no particle identification.
  V0 candidates are opposite-charge track pairs with a vertex fit, kept if: DCA between daughters < 2 cm,
  cos(pointing angle) > 0.9, decay radius > 0.2 cm, and 0.42 < m(pi+ pi-) < 0.58 GeV.
  Signal: true K0S -> pi+ pi- with both daughters reconstructed. Background: everything else that passes, i.e.
  combinatorial pairs of primary and secondary tracks, Lambda and anti-Lambda decays, and other secondaries.
  Every item's labels are half K0S, half background, so log-odds 0 means "as likely K0S as background".

Features (index: name [unit]):
  0 DCA between the two daughter tracks [cm]
  1 cosine of the pointing angle
  2 DCA of the positive daughter to the primary vertex [cm]
  3 DCA of the negative daughter to the primary vertex [cm]
  4 transverse decay radius [cm]
  5 m L / p, the proper decay length using the K0S mass [cm]
  6 Armenteros ratio qT / |alpha|
  7 |m(p pi) - m_Lambda| [GeV], the mass under the Lambda (or anti-Lambda) hypothesis minus 1.11568 GeV

Tree format. A node is {"f": <feature index>, "t": <threshold, physical units>, "lo": <child if feature < t>,
"hi": <child if feature >= t>}; a child is another node or a number (the leaf's log-odds contribution).
The score of a candidate is bias + the sum over trees of the leaf it lands in. At most 12 trees, depth at most 4,
|leaf| <= 20. Fewer, physically motivated trees are fine; leaves on regions that hold almost no training candidates
should stay small, because the boosting will only correct what the labels can see.

Each item is one independent problem (a different label sample; "labels" is how many labelled candidates it has).
For each item you get the 5/25/50/75/95 % quantiles of every feature for the K0S and background candidates of its
training part. From round 2 on you also get your previous ensembles for that item and, on its held-back labels, the
AUC, the background rejection at 95 % K0S efficiency (R95) and the log-loss of your trees alone and of your trees plus
the boosting, how many trees the boosting had to add, its split-gain share per feature and its strongest splits
(feature, threshold): these show what your trees still miss. "training_survivors" are the quantiles of training
background that your trees keep at 95 % training K0S efficiency. Rejection on few held-back labels is noisy: with a few
hundred labels or less, trust physics and the log-loss more than R95. Edit branches (thresholds, leaves, add or remove
nodes and trees) to improve the final selection on an independent test sample; the round with the best held-back
log-loss of trees plus boosting is kept for each item.

Answer: write ONE JSON file whose keys are the item ids of this batch and whose values are
{"note": "<one line: what you changed and why>", "bias": <number>, "trees": [<node>, ...]}.
Treat every item on its own; do not copy one item's numbers into another without reason.

# Round 1. Items in this file: 15

## c_N10000_s0
```json
{"item": "c_N10000_s0", "labels": 10000, "training_labels": 8000, "held_back_labels": 2000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002633, 0.0132, 0.03141, 0.06157, 0.1858], "background": [0.02558, 0.1907, 0.4396, 0.9261, 1.722]}, "1 cos of the pointing angle": {"K0S": [0.9975, 0.9999, 1.0, 1.0, 1.0], "background": [0.9065, 0.9354, 0.9691, 0.9931, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1555, 0.585, 1.342, 2.717, 6.395], "background": [0.02145, 0.02949, 0.06417, 0.7134, 3.383]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1652, 0.5898, 1.339, 2.736, 6.445], "background": [0.02136, 0.02918, 0.05943, 0.6639, 3.615]}, "4 transverse decay radius [cm]": {"K0S": [0.3767, 1.309, 3.392, 7.88, 19.76], "background": [0.2268, 0.428, 0.9789, 2.306, 20.04]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2722, 0.973, 2.036, 3.744, 7.839], "background": [0.1355, 0.2873, 0.6342, 1.529, 9.438]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1412, 0.2635, 0.4597, 0.9962, 2.0], "background": [0.0305, 0.1731, 0.3414, 0.7356, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01438, 0.05616, 0.08817, 0.1, 0.1], "background": [0.001079, 0.03361, 0.07234, 0.1, 0.1]}}}
```

## c_N10000_s1
```json
{"item": "c_N10000_s1", "labels": 10000, "training_labels": 8000, "held_back_labels": 2000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.00269, 0.01401, 0.03144, 0.063, 0.1822], "background": [0.01965, 0.1853, 0.4475, 0.9407, 1.71]}, "1 cos of the pointing angle": {"K0S": [0.9977, 0.9999, 1.0, 1.0, 1.0], "background": [0.9076, 0.9357, 0.968, 0.9925, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1739, 0.5825, 1.347, 2.671, 6.336], "background": [0.02136, 0.02886, 0.06295, 0.6772, 3.207]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1778, 0.5952, 1.315, 2.662, 6.184], "background": [0.02143, 0.02898, 0.06027, 0.6759, 3.327]}, "4 transverse decay radius [cm]": {"K0S": [0.3737, 1.339, 3.315, 7.667, 20.13], "background": [0.227, 0.4279, 0.9911, 2.28, 15.25]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2867, 0.9583, 2.012, 3.728, 7.698], "background": [0.1347, 0.2872, 0.6614, 1.502, 7.751]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1368, 0.2639, 0.4617, 0.9761, 2.0], "background": [0.03042, 0.1635, 0.3211, 0.7038, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01223, 0.05641, 0.08835, 0.1, 0.1], "background": [0.0008462, 0.03147, 0.07068, 0.1, 0.1]}}}
```

## c_N10000_s2
```json
{"item": "c_N10000_s2", "labels": 10000, "training_labels": 8000, "held_back_labels": 2000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002499, 0.0131, 0.03084, 0.06059, 0.1726], "background": [0.02651, 0.2029, 0.4531, 0.915, 1.683]}, "1 cos of the pointing angle": {"K0S": [0.9979, 0.9999, 1.0, 1.0, 1.0], "background": [0.908, 0.9356, 0.9693, 0.9935, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1807, 0.6, 1.299, 2.713, 6.475], "background": [0.02155, 0.02921, 0.06014, 0.6823, 3.364]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1821, 0.6102, 1.333, 2.717, 6.366], "background": [0.02136, 0.02844, 0.05649, 0.6582, 3.196]}, "4 transverse decay radius [cm]": {"K0S": [0.4069, 1.362, 3.268, 7.091, 19.71], "background": [0.2302, 0.427, 0.9638, 2.338, 18.62]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2886, 0.9546, 1.986, 3.81, 7.848], "background": [0.1411, 0.2883, 0.6579, 1.558, 9.382]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1406, 0.2664, 0.4697, 1.025, 2.0], "background": [0.0293, 0.1757, 0.3433, 0.7537, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01472, 0.0583, 0.089, 0.1, 0.1], "background": [0.001035, 0.03382, 0.07299, 0.1, 0.1]}}}
```

## c_N10000_s3
```json
{"item": "c_N10000_s3", "labels": 10000, "training_labels": 8000, "held_back_labels": 2000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.00271, 0.01363, 0.03137, 0.06396, 0.1898], "background": [0.02204, 0.1798, 0.4251, 0.9211, 1.691]}, "1 cos of the pointing angle": {"K0S": [0.9977, 0.9998, 1.0, 1.0, 1.0], "background": [0.9074, 0.9372, 0.9698, 0.9934, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1614, 0.5481, 1.26, 2.618, 6.345], "background": [0.02142, 0.02919, 0.06332, 0.6571, 3.344]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.154, 0.5719, 1.276, 2.652, 6.401], "background": [0.0212, 0.02947, 0.0585, 0.679, 3.243]}, "4 transverse decay radius [cm]": {"K0S": [0.3706, 1.256, 3.175, 7.178, 19.74], "background": [0.2323, 0.4402, 0.9867, 2.414, 19.83]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2543, 0.8879, 1.899, 3.704, 7.79], "background": [0.1319, 0.2976, 0.6721, 1.622, 9.672]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1376, 0.2724, 0.4788, 1.043, 2.0], "background": [0.02766, 0.1638, 0.3364, 0.7096, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01387, 0.05879, 0.08963, 0.1, 0.1], "background": [0.000865, 0.03266, 0.07259, 0.1, 0.1]}}}
```

## c_N10000_s4
```json
{"item": "c_N10000_s4", "labels": 10000, "training_labels": 8000, "held_back_labels": 2000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002346, 0.01284, 0.03073, 0.06237, 0.1896], "background": [0.0249, 0.1951, 0.4294, 0.9217, 1.71]}, "1 cos of the pointing angle": {"K0S": [0.9978, 0.9999, 1.0, 1.0, 1.0], "background": [0.9071, 0.9338, 0.9665, 0.9924, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1673, 0.588, 1.318, 2.722, 6.52], "background": [0.0213, 0.02912, 0.05871, 0.6817, 3.725]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.161, 0.5441, 1.331, 2.724, 6.666], "background": [0.02138, 0.02896, 0.06222, 0.7316, 3.556]}, "4 transverse decay radius [cm]": {"K0S": [0.3776, 1.298, 3.273, 7.355, 19.3], "background": [0.225, 0.4117, 0.996, 2.408, 16.83]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.275, 0.9009, 2.025, 3.791, 8.109], "background": [0.1347, 0.2823, 0.6661, 1.655, 8.749]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.139, 0.2711, 0.4761, 1.012, 2.0], "background": [0.03093, 0.175, 0.3281, 0.7031, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.0137, 0.0583, 0.08901, 0.1, 0.1], "background": [0.001059, 0.03357, 0.07168, 0.1, 0.1]}}}
```

## c_N30000_s0
```json
{"item": "c_N30000_s0", "labels": 30000, "training_labels": 24000, "held_back_labels": 6000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002648, 0.0139, 0.03238, 0.06267, 0.1801], "background": [0.02303, 0.1828, 0.4304, 0.9159, 1.708]}, "1 cos of the pointing angle": {"K0S": [0.9975, 0.9999, 1.0, 1.0, 1.0], "background": [0.9076, 0.9365, 0.9696, 0.9936, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1632, 0.5753, 1.323, 2.685, 6.507], "background": [0.02138, 0.02911, 0.0594, 0.6706, 3.434]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1683, 0.5728, 1.302, 2.68, 6.566], "background": [0.02134, 0.02874, 0.05834, 0.6826, 3.44]}, "4 transverse decay radius [cm]": {"K0S": [0.3756, 1.31, 3.24, 7.344, 19.77], "background": [0.2289, 0.4301, 0.9681, 2.436, 19.36]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2788, 0.9303, 2.008, 3.877, 7.785], "background": [0.1332, 0.2857, 0.6532, 1.594, 9.835]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1377, 0.264, 0.4542, 0.9965, 2.0], "background": [0.03252, 0.1679, 0.3356, 0.7287, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01344, 0.05767, 0.08712, 0.1, 0.1], "background": [0.0009252, 0.03261, 0.07255, 0.1, 0.1]}}}
```

## c_N30000_s1
```json
{"item": "c_N30000_s1", "labels": 30000, "training_labels": 24000, "held_back_labels": 6000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.00242, 0.01336, 0.03097, 0.0609, 0.1682], "background": [0.02414, 0.1854, 0.426, 0.9157, 1.694]}, "1 cos of the pointing angle": {"K0S": [0.9975, 0.9999, 1.0, 1.0, 1.0], "background": [0.9075, 0.9356, 0.9693, 0.9931, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1709, 0.5715, 1.29, 2.682, 6.477], "background": [0.02138, 0.02867, 0.05799, 0.6592, 3.324]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.167, 0.5746, 1.288, 2.648, 6.598], "background": [0.02133, 0.029, 0.05951, 0.665, 3.415]}, "4 transverse decay radius [cm]": {"K0S": [0.3801, 1.264, 3.152, 7.201, 20.15], "background": [0.2288, 0.4242, 0.9552, 2.298, 17.52]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2665, 0.922, 1.981, 3.796, 7.937], "background": [0.1354, 0.2849, 0.6482, 1.545, 8.808]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1417, 0.2658, 0.4656, 1.011, 2.0], "background": [0.03064, 0.1745, 0.34, 0.7354, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01526, 0.05852, 0.08866, 0.1, 0.1], "background": [0.001002, 0.03414, 0.07268, 0.1, 0.1]}}}
```

## c_N30000_s2
```json
{"item": "c_N30000_s2", "labels": 30000, "training_labels": 24000, "held_back_labels": 6000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002537, 0.01357, 0.03137, 0.06261, 0.1854], "background": [0.02254, 0.1826, 0.4296, 0.9199, 1.723]}, "1 cos of the pointing angle": {"K0S": [0.9974, 0.9999, 1.0, 1.0, 1.0], "background": [0.9071, 0.9353, 0.969, 0.9932, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1682, 0.5767, 1.321, 2.727, 6.581], "background": [0.02142, 0.02908, 0.06233, 0.6856, 3.468]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1689, 0.5785, 1.321, 2.692, 6.447], "background": [0.02141, 0.02932, 0.06145, 0.7122, 3.468]}, "4 transverse decay radius [cm]": {"K0S": [0.3768, 1.287, 3.197, 7.372, 19.37], "background": [0.2305, 0.4368, 1.004, 2.428, 18.14]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.27, 0.9346, 2.026, 3.855, 7.959], "background": [0.1369, 0.2981, 0.6819, 1.591, 9.046]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1402, 0.2616, 0.4645, 1.012, 2.0], "background": [0.02789, 0.1663, 0.3343, 0.7435, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01436, 0.05705, 0.08843, 0.1, 0.1], "background": [0.0009293, 0.03241, 0.07242, 0.1, 0.1]}}}
```

## c_N30000_s3
```json
{"item": "c_N30000_s3", "labels": 30000, "training_labels": 24000, "held_back_labels": 6000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.00278, 0.01356, 0.03113, 0.06167, 0.1804], "background": [0.02426, 0.1825, 0.4349, 0.93, 1.723]}, "1 cos of the pointing angle": {"K0S": [0.9976, 0.9999, 1.0, 1.0, 1.0], "background": [0.9074, 0.9365, 0.9692, 0.9935, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1685, 0.5788, 1.323, 2.717, 6.64], "background": [0.02125, 0.02892, 0.06058, 0.6972, 3.416]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1664, 0.5846, 1.326, 2.73, 6.588], "background": [0.02134, 0.02884, 0.05724, 0.6665, 3.361]}, "4 transverse decay radius [cm]": {"K0S": [0.3869, 1.279, 3.269, 7.565, 20.13], "background": [0.2277, 0.4387, 0.9923, 2.319, 16.09]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2677, 0.9371, 2.009, 3.866, 7.954], "background": [0.1351, 0.2954, 0.6622, 1.566, 7.486]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1393, 0.2625, 0.4645, 0.9667, 2.0], "background": [0.03123, 0.1731, 0.3395, 0.763, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01375, 0.05719, 0.08845, 0.1, 0.1], "background": [0.0009473, 0.03362, 0.07342, 0.1, 0.1]}}}
```

## c_N30000_s4
```json
{"item": "c_N30000_s4", "labels": 30000, "training_labels": 24000, "held_back_labels": 6000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002716, 0.01382, 0.03191, 0.06196, 0.1831], "background": [0.02382, 0.1867, 0.4356, 0.9179, 1.71]}, "1 cos of the pointing angle": {"K0S": [0.9974, 0.9999, 1.0, 1.0, 1.0], "background": [0.9079, 0.9364, 0.9689, 0.9933, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1766, 0.5678, 1.293, 2.668, 6.44], "background": [0.0214, 0.02924, 0.05866, 0.6631, 3.338]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1644, 0.5777, 1.325, 2.75, 6.497], "background": [0.02139, 0.02891, 0.05766, 0.6604, 3.334]}, "4 transverse decay radius [cm]": {"K0S": [0.386, 1.289, 3.231, 7.119, 19.79], "background": [0.2296, 0.4321, 0.9634, 2.348, 16.15]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2777, 0.9379, 1.992, 3.821, 7.925], "background": [0.1366, 0.2927, 0.6631, 1.567, 8.455]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1388, 0.2633, 0.4559, 0.9861, 2.0], "background": [0.02788, 0.1681, 0.3364, 0.7325, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01448, 0.05697, 0.08746, 0.1, 0.1], "background": [0.0009194, 0.03327, 0.07276, 0.1, 0.1]}}}
```

## c_N50000_s0
```json
{"item": "c_N50000_s0", "labels": 50000, "training_labels": 40000, "held_back_labels": 10000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002588, 0.01387, 0.03161, 0.0625, 0.1834], "background": [0.02399, 0.1862, 0.4251, 0.9129, 1.708]}, "1 cos of the pointing angle": {"K0S": [0.9976, 0.9999, 1.0, 1.0, 1.0], "background": [0.9081, 0.937, 0.9696, 0.9932, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.166, 0.5725, 1.309, 2.707, 6.61], "background": [0.02141, 0.02912, 0.05882, 0.6732, 3.448]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1657, 0.5729, 1.321, 2.752, 6.45], "background": [0.02137, 0.02881, 0.0588, 0.6823, 3.353]}, "4 transverse decay radius [cm]": {"K0S": [0.3756, 1.273, 3.265, 7.397, 19.53], "background": [0.2279, 0.421, 0.9703, 2.342, 17.5]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2702, 0.9385, 2.028, 3.853, 7.917], "background": [0.1354, 0.283, 0.6583, 1.562, 8.773]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1392, 0.2644, 0.4608, 0.9924, 2.0], "background": [0.02913, 0.1681, 0.3388, 0.7415, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01355, 0.05767, 0.08802, 0.1, 0.1], "background": [0.0009622, 0.03344, 0.0733, 0.1, 0.1]}}}
```

## c_N50000_s1
```json
{"item": "c_N50000_s1", "labels": 50000, "training_labels": 40000, "held_back_labels": 10000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002507, 0.0133, 0.03056, 0.06085, 0.1741], "background": [0.02378, 0.1836, 0.4236, 0.9141, 1.703]}, "1 cos of the pointing angle": {"K0S": [0.9975, 0.9999, 1.0, 1.0, 1.0], "background": [0.908, 0.9367, 0.9693, 0.9932, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1685, 0.5698, 1.31, 2.658, 6.501], "background": [0.02136, 0.02915, 0.05847, 0.6681, 3.331]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1679, 0.572, 1.304, 2.648, 6.351], "background": [0.02138, 0.02917, 0.05872, 0.6637, 3.387]}, "4 transverse decay radius [cm]": {"K0S": [0.3717, 1.291, 3.195, 7.297, 19.74], "background": [0.2274, 0.4254, 0.9758, 2.339, 17.51]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2719, 0.9166, 2.009, 3.818, 7.822], "background": [0.1349, 0.2864, 0.6563, 1.545, 8.924]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1388, 0.2631, 0.4653, 0.9828, 2.0], "background": [0.02856, 0.1655, 0.3356, 0.7331, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01392, 0.05708, 0.0884, 0.1, 0.1], "background": [0.000918, 0.03213, 0.07215, 0.1, 0.1]}}}
```

## c_N50000_s2
```json
{"item": "c_N50000_s2", "labels": 50000, "training_labels": 40000, "held_back_labels": 10000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002608, 0.01376, 0.03138, 0.06197, 0.1849], "background": [0.02437, 0.1891, 0.4315, 0.9173, 1.716]}, "1 cos of the pointing angle": {"K0S": [0.9975, 0.9998, 1.0, 1.0, 1.0], "background": [0.9075, 0.9359, 0.9686, 0.9934, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.167, 0.565, 1.296, 2.655, 6.562], "background": [0.02133, 0.02903, 0.05913, 0.6722, 3.482]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1669, 0.5746, 1.302, 2.689, 6.481], "background": [0.02138, 0.02919, 0.05922, 0.7003, 3.384]}, "4 transverse decay radius [cm]": {"K0S": [0.3673, 1.277, 3.169, 7.364, 19.74], "background": [0.2289, 0.4291, 0.9838, 2.381, 17.82]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2701, 0.9269, 1.993, 3.799, 7.867], "background": [0.1369, 0.2907, 0.6649, 1.574, 8.866]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1413, 0.2657, 0.4625, 0.9841, 2.0], "background": [0.02929, 0.1694, 0.3343, 0.742, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.0136, 0.05804, 0.08819, 0.1, 0.1], "background": [0.0009891, 0.033, 0.07266, 0.1, 0.1]}}}
```

## c_N50000_s3
```json
{"item": "c_N50000_s3", "labels": 50000, "training_labels": 40000, "held_back_labels": 10000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.00262, 0.01379, 0.03155, 0.0625, 0.1887], "background": [0.02442, 0.1865, 0.4323, 0.9217, 1.707]}, "1 cos of the pointing angle": {"K0S": [0.9976, 0.9999, 1.0, 1.0, 1.0], "background": [0.908, 0.9366, 0.969, 0.9932, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1675, 0.5717, 1.302, 2.681, 6.477], "background": [0.02132, 0.02918, 0.06147, 0.6857, 3.429]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1715, 0.5678, 1.288, 2.653, 6.403], "background": [0.02137, 0.02902, 0.06035, 0.6848, 3.394]}, "4 transverse decay radius [cm]": {"K0S": [0.3798, 1.3, 3.218, 7.252, 19.44], "background": [0.2302, 0.4251, 0.9823, 2.323, 17.96]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2743, 0.9214, 1.992, 3.759, 7.8], "background": [0.1355, 0.2882, 0.6654, 1.512, 8.495]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.137, 0.2635, 0.46, 0.9736, 2.0], "background": [0.03202, 0.1715, 0.3401, 0.7538, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01381, 0.05729, 0.08786, 0.1, 0.1], "background": [0.0009452, 0.03257, 0.07277, 0.1, 0.1]}}}
```

## c_N50000_s4
```json
{"item": "c_N50000_s4", "labels": 50000, "training_labels": 40000, "held_back_labels": 10000, "training_quantiles_5_25_50_75_95": {"0 DCA between the two daughter tracks [cm]": {"K0S": [0.002722, 0.01351, 0.03143, 0.06263, 0.1838], "background": [0.02478, 0.1902, 0.4374, 0.9286, 1.727]}, "1 cos of the pointing angle": {"K0S": [0.9976, 0.9999, 1.0, 1.0, 1.0], "background": [0.9073, 0.9364, 0.9688, 0.9932, 1.0]}, "2 DCA of the positive daughter to the PV [cm]": {"K0S": [0.1664, 0.579, 1.329, 2.7, 6.58], "background": [0.02129, 0.0287, 0.05728, 0.6667, 3.471]}, "3 DCA of the negative daughter to the PV [cm]": {"K0S": [0.1698, 0.5771, 1.319, 2.738, 6.551], "background": [0.02136, 0.02915, 0.05915, 0.6666, 3.377]}, "4 transverse decay radius [cm]": {"K0S": [0.3785, 1.317, 3.269, 7.431, 19.93], "background": [0.2298, 0.4291, 0.9708, 2.339, 18.05]}, "5 m L / p, proper decay length with the K0S mass [cm]": {"K0S": [0.2677, 0.9377, 2.014, 3.815, 7.915], "background": [0.1354, 0.2878, 0.6586, 1.56, 9.138]}, "6 Armenteros ratio qT / |alpha|": {"K0S": [0.1415, 0.2648, 0.4672, 0.9953, 2.0], "background": [0.03007, 0.173, 0.3446, 0.7474, 2.0]}, "7 |m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)": {"K0S": [0.01383, 0.05705, 0.08881, 0.1, 0.1], "background": [0.000978, 0.03442, 0.07385, 0.1, 0.1]}}}
```
