# TV-SPEC-00 — Spectral Ablation Protocol Preregistration

- Status: **NOT_EVALUATED**
- Modality: Sentinel-2 10 m Multiband (B2, B3, B4, B8)
- Architecture: 256 inputs $\to$ 24 hidden $\to$ $N$ classes
- Principle: Controlled spectral ablation without dimensional reduction or parameter changes.

---

## 1. Causal Question

> *Which spectral bands (B2, B3, B4, B8) contribute discriminative signal to class separation in natural land cover classification, and what is the marginal degradation when individual bands are neutralized?*

---

## 2. Experimental Controls & Invariants

To avoid confounding **spectral information loss** with **neural network capacity changes**, all experimental conditions must strictly preserve:

1. **Fixed Architecture**: Exactly 256 inputs ($8 \times 8 \times 4$), 24 hidden neurons, $N$ output classes.
2. **Fixed Parameter Count**: No reduction of weights or biases across conditions.
3. **Fixed Spatial Sampling**: Exact same ROIs, patch origins $(x, y)$, and geometric supports.
4. **Fixed Split Partition**: Same sample assignment to TRAIN, DEV, and PROBE splits.
5. **Fixed Optimization Schedule**: Same seed, learning rate, and epoch trajectory.

---

## 3. Comparison Conditions

| Condition ID | Active Bands | Neutralized Band | Input Dimensions | Capacity (Params) |
|---|---|---|---|---|
| `FULL` | B2, B3, B4, B8 | None | 256 | Identical |
| `MASK-B2` | B3, B4, B8 | B2 (Blue) $\to 0.0$ | 256 | Identical |
| `MASK-B3` | B2, B4, B8 | B3 (Green) $\to 0.0$ | 256 | Identical |
| `MASK-B4` | B2, B3, B8 | B4 (Red) $\to 0.0$ | 256 | Identical |
| `MASK-B8` | B2, B3, B4 | B8 (NIR) $\to 0.0$ | 256 | Identical |

---

## 4. Evaluation Protocol

- Primary Metric: PROBE Generalization Accuracy & Cross-Entropy Loss.
- Secondary Metric: Spatial Margin Distribution on Dense Inference.
- Decision Rule: A band is considered a major signal contributor if neutralizing it produces a statistically significant drop in DEV/PROBE accuracy compared to the `FULL` control condition under identical seeds.

---

## 5. Current Evidence Status

- Protocol Template: **READY**
- Infrastructure: **READY**
- Execution: **NOT_EVALUATED** (no scientific claims are made prior to authorized execution).
