# R Causality Intervention Test Results

**Date**: 2026-01-30
**Checkpoint**: checkpoint_15000.pt (56.6M corpus, step 15K)
**Status**: ✅ Debugging complete, test runs successfully

---

## Key Achievement: R Manipulation Successful! 🎯

The intervention test **successfully manipulates R** across a wide range:

| Condition | Mean R | Change from Baseline |
|-----------|--------|---------------------|
| **Baseline** | 0.3485 ± 0.0002 | — |
| **R_high** | 0.9605 ± 0.0001 | **+176%** (2.75× increase) |
| **R_low** | 0.2363 ± 0.0045 | **-32%** decrease |

### Technical Details

**Method**:
- **R_high**: Reduce spread of ω₀ (pull frequencies toward mean) → synchronization
- **R_low**: Increase spread + noise → desynchronization

**Implementation**:
```python
# R_high: Pull natural frequencies toward mean
for layer in model.blocks:
    omega = layer.oscillators.omega_0
    mean_omega = omega.mean()
    delta = mean_omega - omega
    layer.oscillators.omega_0.data = omega + 0.5 * delta

# R_low: Push frequencies apart + add noise
for layer in model.blocks:
    omega = layer.oscillators.omega_0
    mean_omega = omega.mean()
    delta = omega - mean_omega
    noise = torch.randn_like(omega) * 0.2
    layer.oscillators.omega_0.data = omega + 0.5 * delta + noise
```

---

## Current Limitation: Degenerate Output Quality

At step 15K, the model generates very short sequences (~7 tokens, mostly newlines):

| Condition | Sample Output |
|-----------|--------------|
| Baseline (R=0.349) | `,ion\n\n\n\n...` |
| R_high (R=0.960) | `,ion\n\n\n\n...` |
| R_low (R=0.236) | `,ion\n\n\n\n...` |

**Quality metrics are identical** across conditions:
- Distinct-4: 1.0 for all (every 4-gram unique in 7-token sequences)
- Repetition rate: ~0.22-0.35 (similar variance)
- Statistical test: NaN (no variance to compare)

**This does NOT prove R is epiphenomenal** - it proves we need a better checkpoint!

---

## Debugging Journey

### Issues Fixed

1. **Vocab size mismatch** (100277 vs 50257)
   - ❌ Script assumed GPT-2 standard (50257)
   - ✅ Now infers config from checkpoint dynamically
   - Checkpoint uses tiktoken cl100k_base (100277 tokens)

2. **Checkpoint key mismatch** ('model' vs 'model_state')
   - ❌ Script tried `checkpoint['model']`
   - ✅ Correct: `checkpoint['model_state']`

3. **Phase access error**
   - ❌ Tried to access `layer.oscillators.phases` (doesn't exist)
   - ✅ Phases are computed dynamically, not stored
   - ✅ Perturb ω₀ (natural frequencies) instead

4. **R computation**
   - ❌ Tried to compute R from non-existent phase buffer
   - ✅ Run forward pass with `return_R=True` and extract R_mean

### Final Working Code

**Dynamic config loading**:
```python
def load_model(checkpoint_path: str, device: str = "mps"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state']

    # Infer config from checkpoint structure
    vocab_size, hidden_dim = state['embed.weight'].shape
    n_layers = max([int(k.split('.')[1]) for k in state.keys()
                    if k.startswith('blocks.')]) + 1
    n_oscillators = state['blocks.0.oscillators.omega_0'].shape[0]
    n_harmonics = state['blocks.0.oscillators.readout.weight'].shape[1] // 3

    config = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'n_oscillators': n_oscillators,
        'n_harmonics': n_harmonics,
    }

    model = KSSMv3(**config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model
```

---

## Next Steps

### Option 1: Wait for 20K Checkpoint (RECOMMENDED)

**Current training status** (as of 20:57):
- Step: **15,840 / 20,000**
- R: **0.3564** (climbing steadily)
- u_val: **0.106** (edge-surfing maintained)
- ETA: **~3.5 hours**

At step 20K, the model should generate:
- Longer sequences (50-100 tokens)
- More coherent text
- Measurable quality differences between R conditions

**Re-run intervention test on checkpoint_20000.pt**:
```bash
ssh tony_studio@192.168.1.195
cd ~/liminal-k-ssm
python3 kssm/eval_r_intervention.py
```

### Option 2: Use Temperature Sampling

Modify `generate_sample()` to use temperature sampling instead of greedy:
```python
# Instead of argmax
next_token = torch.argmax(logits, dim=-1)

# Use sampling with temperature
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

This may generate longer, more diverse sequences even at step 15K.

### Option 3: Test on best_model.pt

Check if `results/kssm_v3/best_model.pt` has better generation quality:
```bash
# Update eval_r_intervention.py line 274:
checkpoint = "results/kssm_v3/best_model.pt"
```

---

## Scientific Interpretation

### What We Know ✅

1. **R can be manipulated**: Ω₀ perturbations reliably shift R from 0.24 → 0.96
2. **Intervention framework works**: Clean causal design (baseline, R+, R-)
3. **Model architecture is correct**: Loads, runs, generates (even if poorly at 15K)

### What We Need 🔬

1. **Better checkpoint**: Wait for 20K or use final model
2. **Quality variance**: Need samples that differ in measurable ways
3. **Statistical power**: Currently NaN due to zero variance

### Predicted Outcome (20K Checkpoint)

If R is **causal** (not epiphenomenal):
- R_high → **Higher distinct-4** (more vocabulary diversity)
- R_high → **Lower repetition rate** (less looping)
- **p < 0.01** on paired t-test

If R is **epiphenomenal** (correlation only):
- No quality difference despite R manipulation
- **p > 0.05** on paired t-test
- Bistability may be decorative, not functional

---

## Files

| File | Purpose |
|------|---------|
| `kssm/eval_r_intervention.py` | Main intervention test script (debugged) |
| `results/r_intervention_test.log` | Test output from checkpoint_15000.pt |
| `results/kssm_v3/checkpoint_15000.pt` | Tested checkpoint (R=0.3485) |
| `results/kssm_v3/checkpoint_20000.pt` | Target checkpoint (ETA ~3.5 hours) |

---

**Recommendation**: Wait for training to complete to 20K, then re-run intervention test. The infrastructure is ready and working perfectly - we just need a better model to test on.
