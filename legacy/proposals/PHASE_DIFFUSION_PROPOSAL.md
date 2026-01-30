# Phase-Diffusion: Kuramoto Oscillators in Discrete Diffusion Language Models

## Why Diffusion Fixes the Phase-Mamba Problem

### The Core Issue with Phase-Mamba

```
Layer 32: h' = h * (1 + α*R)  ← Modulation happens ONCE
Layer 33-63: LayerNorm(...)   ← Washed out
Output: No trace of R
```

**Result**: R is epiphenomenal. It computes something but doesn't affect generation.

### How Diffusion Solves This

```
t=1.0 (fully masked) → Denoise step 1 → step 2 → ... → step 100 → t=0.0 (clear)
                       ↑               ↑              ↑
                       R influences    R influences   R influences
                       remask          remask         remask
```

**Key insight**: In diffusion models, R can influence *every denoising step*. The iterative process means:
1. No single LayerNorm can wash it out
2. Effects accumulate across steps
3. The denoising process IS dynamics (Ṙ ≠ 0 by construction)

---

## Available Diffusion LLMs

| Model | Size | Feasibility on Mac Studio |
|-------|------|---------------------------|
| LLaDA 8B | 8B | Borderline (needs quantization) |
| LLaDA-MoE | 7B (1B active) | Possibly |
| **MDLM** | **130M** | **Easy - GPT2-medium size** |
| dLLM | Various | Research code |

**Recommendation**: Start with MDLM. It's small enough to iterate quickly, and the architecture is well-documented.

---

## The Phase-Diffusion Architecture

### Masked Diffusion Process

MDLM uses a simple forward process:
- t=1.0: All tokens masked with [MASK]
- t=0.5: 50% of tokens masked
- t=0.0: No tokens masked (clean text)

Denoising reverses this: predict masked tokens, gradually unmask.

### Phase Core Integration Points

```python
class PhaseDiffusion:
    """Phase Core integrated with masked diffusion."""

    def __init__(self, base_model, n_oscillators=16, K=2.0):
        self.model = base_model
        self.phase_core = KuramotoPhaseCore(n_oscillators, K)

    def denoise_step(self, x_t, t, mask):
        """Single denoising step with Phase Core influence."""

        # 1. Get model predictions for masked tokens
        hidden = self.model.encode(x_t)
        logits = self.model.decode(hidden)

        # 2. Compute R from hidden states
        R = self.phase_core(hidden)

        # 3. R-GUIDED REMASKING (the key innovation)
        # High R → confident, unmask more tokens
        # Low R → uncertain, keep more masked

        confidence = torch.softmax(logits, dim=-1).max(dim=-1).values

        # Modulate confidence by R
        adjusted_confidence = confidence * (0.5 + 0.5 * R)

        # Select tokens to unmask based on adjusted confidence
        n_unmask = int((1 - t) * seq_len * (0.8 + 0.4 * R))
        unmask_indices = adjusted_confidence.topk(n_unmask).indices

        # 4. Update mask
        new_mask = mask.clone()
        new_mask[unmask_indices] = False

        # 5. Sample tokens for unmasked positions
        x_next = sample_from_logits(logits, ~new_mask)

        return x_next, new_mask, R
```

### Why R Can Now Be Causal

| Phase-Mamba | Phase-Diffusion |
|-------------|-----------------|
| R computed once | R computed at every step |
| Modulation washed out | R directly controls unmasking |
| No path to output | R → unmask decision → output |
| Static at inference | Dynamic across denoising |

---

## Experimental Protocol

### Experiment 1: Baseline Diffusion vs Phase-Diffusion

Compare:
- **MDLM baseline**: Standard confidence-based unmasking
- **Phase-MDLM**: R-guided unmasking

Metrics:
- Perplexity (does it still generate coherent text?)
- Entropy distribution across denoising steps
- Distinct-n (diversity)

### Experiment 2: R Dynamics During Denoising

Track R across all denoising steps:

```
t=1.0 (all masked):   R = ?
t=0.75:               R = ?
t=0.50:               R = ?
t=0.25:               R = ?
t=0.0 (unmasked):     R = ?
```

**Hypothesis**: R should vary across steps (Ṙ ≠ 0), unlike Phase-Mamba where it collapsed.

### Experiment 3: Intervention

Force R to specific values during denoising:
- R=0.3 (cautious): Should unmask slowly, higher diversity
- R=0.7 (confident): Should unmask quickly, lower diversity
- R=free: Natural dynamics

**If intervention changes output**: R is causal
**If no effect**: Still disconnected (but this is less likely with direct control of unmasking)

---

## Implementation Plan

### Phase 1: Minimal MDLM on MPS

1. Extract core MDLM architecture (remove flash_attn dependency)
2. Implement simple transformer with standard attention
3. Verify basic masked diffusion works on Mac Studio

### Phase 2: Add Phase Core

1. Graft Phase Core onto transformer hidden states
2. Implement R-guided remasking
3. Train on small corpus (WikiText or similar)

### Phase 3: Experiments

1. Baseline comparison
2. R dynamics tracking
3. Intervention experiments

---

## Why This Should Work

1. **Direct causal path**: R → unmask decision → which tokens appear
   - No LayerNorm can wash this out because it's a *decision*, not a modulation

2. **Built-in dynamics**: Denoising steps naturally vary t from 1→0
   - The system MUST change over time
   - Ṙ ≠ 0 by construction

3. **Uncertainty is architectural**: Masks = uncertainty
   - U doesn't need artificial regulation
   - The diffusion process handles it

4. **Testable causality**: We can measure if R-guided remasking produces different outputs than random remasking

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| MPS incompatibility | Medium | Use simple attention, no flash_attn |
| R still doesn't matter | Low | Direct control of unmasking should work |
| Poor text quality | Medium | Start with pretrained MDLM weights |
| Too slow | High | Diffusion IS slow; accept for research |

---

## Next Steps

1. **Create MPS-compatible MDLM**
   - Port minimal architecture
   - Test basic generation

2. **Integrate Phase Core**
   - Hook at transformer middle layer
   - Implement R-guided remasking

3. **Run experiments**
   - Compare to baseline
   - Track R dynamics
   - Test intervention

---

## References

- MDLM: https://github.com/kuleshov-group/mdlm
- LLaDA: https://github.com/ML-GSAI/LLaDA
- Phase-Mamba null result: This repository

---

*The spiral continues. Diffusion models breathe by design.* 🌀
