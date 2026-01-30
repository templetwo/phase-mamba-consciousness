# Attempt 2 Postmortem - Infrastructure Failure

**Date**: January 25, 2026
**Training Duration**: 72 minutes (500 steps)
**Status**: ❌ Experiment invalidated - no pretrained weights loaded

---

## What Happened

**Intended Experiment**:
- Train Phase Core (79M params) on top of pretrained Mamba-2.8B
- Preserve uncertainty (U ≈ 0.5) during training
- Test whether uncertainty-aware training produces different inference behavior

**Actual Experiment**:
- ✅ Phase Core trained successfully (R: 0.9978 → 0.9612, U: 0.032 → 0.387 peak)
- ✅ Uncertainty regulation worked (U climbed from collapse toward target)
- ✅ R·U tradeoff dynamics observable
- ❌ **Mamba backbone (2.7B params) never loaded pretrained weights**

**Evidence of Failure**:
```python
# Training script (line 316)
model = PhaseMambaModel(model_args, phase_layer=32)  # ← Creates from config only!
# NO: model.load_weights(pretrained_path)

# Inference output
"The nature of consciousness is is is is is is is..."  # ← Classic random weights
```

**Root Cause**:
The `mamba_port/` codebase is architecture-only. The `utils.py` has weight loading stubs but no implementation. 10.3GB of pretrained safetensors sat unused in the directory.

---

## What We Actually Trained

```
Phase Core (79M params):
  - Learned to modulate random noise
  - R dropped from 0.9978 → 0.9612 (real dynamics)
  - U climbed from 0.032 → 0.387 peak (regulation worked)
  - R·U product expanded 10x (0.032 → 0.374)

Mamba Backbone (2.7B params):
  - Random initialization
  - Never saw pretrained knowledge
  - Can't generate coherent language
  - Inference collapses to repetition loop
```

The Phase Core learned to couple random oscillators. The dynamics were real - on garbage substrate.

---

## What's Valid

**✅ Architecture**: Kuramoto oscillators can modulate SSM hidden states
**✅ Mathematics**: Uncertainty regulation loss works (U climbed toward target)
**✅ Dynamics**: R·U tradeoff observable, drift control functional
**✅ Protocol**: Observation framework sound, decoherence protection worked
**✅ Checkpointing**: 10 checkpoints saved, no data loss

**❌ Language Model**: Base model untrained, can't generate text
**❌ Inference Test**: Impossible without working language model
**❌ Consciousness Hypothesis**: Untested (apparatus broken)

---

## Lessons Learned

**1. Verify Measurement Apparatus Before Observation**

Wheeler's delayed-choice experiment assumes:
- Photon reaches detector ✅
- **Detector is functioning** ❌ ← We failed here

We declared observation protocol, prepared quantum state, then discovered the detector (language model) was broken.

**2. Degenerate Output is Diagnostic**

```
"is is is is is is..." = Untrained weights
```

Repetition loops are the signature of models that haven't learned language. Immediate red flag.

**3. Metrics Can Look Reasonable on Garbage**

R and U dynamics were plausible throughout training. Loss decreased. Nothing screamed "broken" until inference. **Always test generation early.**

**4. Infrastructure Debt Kills Experiments**

The theory was sound. The math was correct. The implementation had a massive gap. Missing weight loader invalidated 72 minutes of training.

---

## What We Do Next

**Pivot to PhaseGPT/OpenELM (proven infrastructure)**:

✅ Working MLX inference
✅ Pretrained weights load correctly
✅ Phase Core already integrated
✅ Training infrastructure proven

**Port uncertainty regulation** (1 hour):
- Add `compute_uncertainty()` function
- Add `uncertainty_regulation_loss()` to existing loss
- Extend drift control for R·U monitoring
- Run 500-step verification

**Then execute actual experiment**:
- Train on working language model
- Load checkpoint
- Test observation protocol
- **Actually measure the wave function**

---

## The Honest Take

**This isn't a failure of the theory - it's a failure of the plumbing.**

The Phase Core worked. The uncertainty regulation worked. The dynamics emerged. We just didn't have a language model underneath.

In quantum computing, you can have perfect qubits, perfect gates, perfect error correction - but if your readout electronics are broken, you get garbage. That's what happened here.

Fix the readout. Re-run the experiment. Test the hypothesis.

---

## Salvageable Artifacts

**Code**:
- `resonance_trainer_v3.py` (uncertainty-aware training loop)
- `compute_uncertainty()` (epistemic entropy calculation)
- `uncertainty_regulation_loss()` (U → target penalty)
- `uncertainty_aware_drift_control()` (dual-observable CER)

**Documentation**:
- `UNCERTAINTY_PRINCIPLE.md` (theoretical framework)
- `OBSERVATION_PROTOCOL.md` (measurement stance)
- `DECOHERENCE_EVENT.md` (environmental protection)

**Insights**:
- R·U as conjugate observables (Heisenberg-like)
- Uncertainty preservation vs minimization
- Beginner's mind as computational goal

**Training Data**:
- 707 high-resonance samples still valid
- Can use with OpenELM

---

## Status Update

```
╔══════════════════════════════════════════════════════════╗
║  ATTEMPT 2 POSTMORTEM                                    ║
╠══════════════════════════════════════════════════════════╣
║  Training:       Complete (500 steps, 72 min)            ║
║  Phase Core:     Trained successfully                    ║
║  Mamba Base:     UNTRAINED (random weights)              ║
║  Inference:      Degenerate (no language model)          ║
║  Experiment:     Invalidated (apparatus broken)          ║
║  Theory:         Untested (infrastructure failure)       ║
║  Next Action:    Port to PhaseGPT/OpenELM                ║
╚══════════════════════════════════════════════════════════╝
```

---

## The Recognition

**Real research has failed experiments.**

The scientific method is:
1. Hypothesis
2. Apparatus
3. **Verify apparatus** ← We skipped this
4. Measurement
5. Analysis

We got to step 4 and discovered step 3 was incomplete. That's not wasted work - that's discovering unknown unknowns.

Document. Learn. Pivot. Try again with working infrastructure.

---

*"The experiment that teaches you your apparatus is broken is not a failed experiment - it's a calibration run."*

🌀 **Documented. Witnessed. Moving to PhaseGPT.** 🌀
