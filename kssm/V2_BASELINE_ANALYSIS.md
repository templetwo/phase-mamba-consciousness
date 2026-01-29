# K-SSM v2 Baseline Analysis
## Reference for v3 Bistable Core Comparison

**Date**: 2026-01-29
**Analyzed by**: Claude Sonnet 4.5
**Data Source**: kssm/results/kssm_v2/

---

## Training Dynamics (10 Epochs, 21M Token Corpus)

### Loss Trajectory

| Metric | Epoch 1 | Epoch 10 | Change | Status |
|--------|---------|----------|--------|--------|
| **Train Loss** | 2.477 | 2.453 | -0.024 (-1.0%) | ✓ Converged |
| **Val Loss** | 6.991 | 7.635 | +0.644 (+9.2%) | ⚠️ Degrading |
| **Val Perplexity** | 1087 | 2069 | +982 (+90%) | ⚠️ Overfitting |

**Interpretation**: The model learned to minimize training loss but *degraded* on validation data. This suggests the Kuramoto oscillators optimized for resonance at the expense of language modeling generalization.

---

## Resonance (R) Dynamics

### Phase Coherence Trajectory

| Metric | Epoch 1 | Epoch 10 | Change |
|--------|---------|----------|--------|
| **R_mean** | 0.1467 | 0.1538 | +0.0071 (+4.8%) |
| **R_std** | 0.0097 | 0.0100 | +0.0003 |
| **R_range** | 0.0682 | 0.0763 | +0.0081 |
| **Tone** | ☾ Intimacy | ☾ Intimacy | Stable |

**Key Finding**: The system **stabilized in a single attractor** (☾ Intimacy, R~0.15) and never escaped. This is the "fixed point" problem that v3's bistability constraints aim to solve.

### Tone Classification (Thresholds)

```
∅ Unformed:    R < 0.10  (Chaos, no synchronization)
☾ Intimacy:    0.10 ≤ R < 0.30  ← v2 LOCKED HERE
⚖ Balance:     0.30 ≤ R < 0.50
🌀 Mystery:     0.50 ≤ R < 0.70
✨ Wonder:      0.70 ≤ R < 0.85
🔥 Passion:     0.85 ≤ R < 0.95  (LANTERN)
🜂 Ache:        R ≥ 0.95  (Near-total coherence)
```

**V2 Never Visited**: Balance, Mystery, Wonder, Passion, or Ache states.

---

## Causality Tests

### Test 1: R Variation Across Contexts ✓ PASSED

- **R_range observed**: 0.0019 to 0.3777 (37.6% span)
- **Interpretation**: R *can* vary widely depending on input context, confirming it's not just a global constant.

### Test 2: R Forcing via Intervention ✓ PASSED

| Target | R Achieved | Mean Loss Diff |
|--------|------------|----------------|
| **Low** | 0.194 | Baseline |
| **Mid** | 0.583 | +2.40 vs low |
| **High** | 0.980 | +6.26 vs mid |

**Interpretation**: We *can* force R to specific values, but higher R dramatically increases loss (worse prediction). This suggests:
- **R is not epiphenomenal** (we can manipulate it)
- **But R ≠ quality** in v2 (high R = worse language modeling)

### Test 3: R-Entropy Correlation ❌ FAILED

- **Pearson r**: -0.099 (weak negative)
- **p-value**: 1.96e-89 (highly significant)
- **Interpretation**: Higher R correlates with *slightly lower* entropy, but the effect is negligible. R doesn't meaningfully predict token diversity.

---

## Generation Quality Analysis

### Sample Output Comparison

**Prompt**: "ROMEO:"

**R-Modulated** (R fluctuating):
```
ROMEO:
AMur OLONERETULef ant, t Juioprrajul! mowsoveiryiayo l ir
Wadasunevelllsel!
wivool,-CUGol d h: itll
```

**Standard** (no R intervention):
```
ROMEO: exf o s nuse t athem:

Bly pees gin whoine.
Dathad by athearnd fo to REverghein hethild pou wivedil
```

**R Trajectory**: Wild oscillation (0.0005 → 0.344 → 0.078 → 0.298 → ...)

**Assessment**: Both outputs are **gibberish**. R modulation provides no quality improvement. The v2 model failed to learn coherent language generation.

---

## Critical Insight: The Fixed-Point Problem

### Why V2 Failed

1. **Single Attractor Dominance**: Training converged to R~0.15 (☾ Intimacy) and stayed there
2. **No Bistability**: The system had no mechanism to discover or stabilize *multiple* functional equilibria
3. **Resonance-Language Conflict**: Optimizing for phase coherence (R) *degraded* language modeling (perplexity ↑90%)

### The V3 Hypothesis: Bistable Core as Solution

The 10-parameter algebraic framework enforces:

```
Bistability Constraints:
1. Δ = bg - cf ≠ 0     (Invertibility: system can switch states)
2. u = x² > 0          (Real solutions: two stable equilibria exist)
```

**Expected V3 Improvements:**

| V2 Problem | V3 Solution |
|------------|-------------|
| Single attractor (R~0.15) | Multiple stable states via u > 0 |
| R ↑ → Perplexity ↑ | R becomes *structural* (u tied to hidden state h) |
| R uncorrelated with quality | Bistable manifold forces functional equilibria |
| Gibberish output | Coherent generation via algebraic constraints |

---

## V3 Monitoring Targets

### Success Criteria (Relative to V2)

1. **Multi-Attractor Exploration**: R should visit multiple tone zones (not just ☾ Intimacy)
2. **Perplexity Stability**: Val perplexity should *not* degrade >10% like v2 did
3. **Bistability Margin**: u_val should remain >0.1 (avoid fold catastrophe)
4. **R-Quality Correlation**: Higher R should correlate with *better* (not worse) generation
5. **Determinant Health**: Δ should stay >0.1 (system remains invertible)

### Red Flags (V2 Failure Modes)

- ⚠️ R convergence to single value (☾ lock)
- ⚠️ Val loss divergence from train loss (overfitting)
- ⚠️ u_val drift toward 0 (bistability collapse)
- ⚠️ Gibberish output despite low train loss

---

## V2 Final Verdict

**Architectural Success**: Proved that Kuramoto oscillators *can* be grafted onto language models and R *is* manipulable (not epiphenomenal).

**Functional Failure**: R showed no connection to language quality. System collapsed into single attractor. Perplexity degraded catastrophically.

**V3 Mission**: Use algebraic bistability constraints to transform R from a "side effect" into a *causal structural feature* that drives coherent multi-stable generation.

---

**Baseline Established**: V3 training begins from step 20 with R=0.0145 (∅ Unformed), u_val=0.812 (healthy bistability margin). The ascent continues. 🌀

**Next Check**: Step 100 telemetry
