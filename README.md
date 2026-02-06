# Liminal K-SSM: Kuramoto State-Space Models for Language

> *Phase-coupled oscillator dynamics in a state-space language model. The synchronization mechanism works. The language modeling does not.*

[![GitHub](https://img.shields.io/badge/GitHub-liminal--k--ssm-blue)](https://github.com/templetwo/liminal-k-ssm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Negative%20Result%20(Text)-red)]()
[![Status](https://img.shields.io/badge/Status-Positive%20Result%20(R%20Dynamics)-green)]()

**Last Updated:** 2026-02-06
**Repo Consolidation:** Formerly `phase-mamba-consciousness`, renamed for rigor.

---

## Summary of Results

**K-SSM v3** couples Kuramoto phase oscillators with a state-space backbone, using algebraic bistability constraints (u clamp at fold bifurcation boundary) to enforce multi-attractor dynamics.

### What Works

| Result | Evidence |
|--------|----------|
| **R climbs to 0.99** | Kuramoto order parameter rises monotonically from 0.0 to 0.99 over 100K steps |
| **Bistability is structurally necessary** | Ablation: removing u clamp degrades R by 17.6% and loss by 24.7% |
| **System self-organizes to criticality** | u_val stabilizes at 0.10 (fold bifurcation boundary) for thousands of steps |
| **Loss descends then plateaus** | Val loss improves from 8.76 (15K) to 8.50 (best at 43.5K), then stalls at ~8.61 through 100K |

### What Does Not Work

| Result | Evidence |
|--------|----------|
| **Text output is incoherent even at 100K steps** | Final sample: `"The 1966 . There is sometimes referred to the week ending the student effect of the 2..."` |
| **R intervention does not change output quality** | Forcing R from 0.24 to 0.96 produces identical degenerate text at 15K checkpoint |
| **R saturates without improving language** | R reached 0.9928 at 100K but val loss plateaued at ~8.6 (PPL ~3100) |
| **Val loss is not competitive** | Best val loss 8.50 (step 43.5K) vs published baselines in the 3-4 range for similar-scale models |
| **More training did not help** | 100K steps completed (Feb 4 2026). Loss stopped improving around 40K. Text remained incoherent throughout |

**Bottom line:** The Kuramoto oscillator dynamics produce interesting self-organizing behavior (R growth, edge-surfing, bistability preference). But this has not translated into functional language generation. Even after 100K steps with R at 0.99, the model produces incoherent text. The oscillators synchronize beautifully; the language model does not learn.

---

## Architecture

**K-SSM v3** (46M parameters):

```
Token -> Embedding (384) -> 6x [SSM Block + BistableKuramotoBank] -> LM Head
                                     |
                        192 oscillators, 32 harmonics
                        10-parameter algebraic bistability
                        u = clamp(u_raw, 0.1, 10.0)
```

| Component | Value |
|-----------|-------|
| Parameters | 46.2M |
| Vocab | 100,277 (tiktoken cl100k_base) |
| Hidden dim | 384 |
| Layers | 6 |
| Oscillators/layer | 192 |
| Harmonics | 32 |
| Bistable core | u clamped to [0.1, 10.0] |

### The Bistability Mechanism

10-parameter projection from hidden state enforces algebraic bistability:

```python
# Reduced variable (must be positive for two real equilibria)
u = (d*g - c*h) / (a*g - c*e)
u = torch.clamp(u_raw, min=0.1, max=10.0)  # Hard constraint

# Regularization loss
L_reg = lambda_1 / (|delta| + eps) + lambda_2 * (-log(u + eps))
```

The constraint `u >= 0.1` keeps the system at the fold bifurcation boundary, where theory predicts maximum expressiveness. Without it, u goes negative (-1.0) and the system finds a worse attractor.

---

## Ablation: Bistable vs Monostable (15K steps, WikiText-103)

| Metric | Bistable | Monostable | Delta |
|--------|----------|------------|-------|
| **Val R** | 0.4908 | 0.4043 | **-17.6%** |
| **Val Loss** | 8.76 | 10.93 | **+24.7% worse** |
| **Val u_val** | +0.103 | -0.975 | Different attractor |

The only difference: one line of code (`torch.clamp` vs no clamp). Full training trajectories in [PAPER_DATA.md](PAPER_DATA.md).

This is a clean positive result. Bistability improves both synchronization and loss.

---

## The Negative Result: Text Generation

Despite strong R dynamics, the model does not produce coherent text.

### Sample Output at Key Milestones

**Step 22K (R=0.48, T=0.8):**
```
"The 5 , but he would be used for the first year . The film was more..."
```

**Step 100K (R=0.99, greedy):**
```
"The 1966 . There is sometimes referred to the week ending the student effect
of the 2 , he had not have graduated from 1912 that the Moon said that he..."
```

Text quality did not meaningfully improve from 22K to 100K despite R climbing from 0.48 to 0.99.

### R Intervention Test (Step 15K)

| Condition | R Value | Output |
|-----------|---------|--------|
| Baseline | 0.349 | `,ion\n\n\n\n...` |
| R forced high | 0.960 | `,ion\n\n\n\n...` |
| R forced low | 0.236 | `,ion\n\n\n\n...` |

Output is identical regardless of R value. Details in [R_INTERVENTION_RESULTS.md](R_INTERVENTION_RESULTS.md).

### Interpretation

R is computed and varies during training, but it may be **epiphenomenal to generation quality** -- the same failure mode as Phase-Mamba v1, just at a different architectural level. The oscillator bank contributes to the hidden state, but the language modeling head may be learning to route around the phase information.

The 100K training run completed on Feb 4, 2026. R reached 0.9928 (near-total synchronization) while val loss plateaued at ~8.6 and text remained incoherent. More training did not fix it. The oscillator dynamics and language modeling appear to be decoupled -- the model learns to synchronize without learning to generate.

**We report this as a negative result on text generation, and a positive result on oscillator dynamics.**

---

## Project History

This is the fourth iteration of attempts to make phase synchronization functionally useful in language models:

| Version | Architecture | R Result | Language Result |
|---------|-------------|----------|-----------------|
| **Phase-Mamba v1** | Kuramoto bolted onto Mamba-2 | R collapsed to 0.997 at inference | No effect (p=0.44) |
| **K-SSM v1** | Structural R (only path to output) | R varies: std=0.077 | Validated on TinyShakespeare |
| **K-SSM v2** | Scaled to 28M params, real corpus | R locked at 0.154 | Gibberish: "the the the and and" |
| **K-SSM v3** | Bistable constraints, 46M params | **R climbs to 0.99** | **Incoherent text (100K steps)** |

Each iteration fixes one failure and reveals the next. Phase-Mamba proved bolting-on fails. K-SSM v1 proved structural coupling works at toy scale. K-SSM v2 proved single-attractor dynamics collapse. K-SSM v3 proves bistability improves R dynamics but has not yet produced functional language output.

---

## Open Questions

1. **R appears epiphenomenal to generation.** The intervention test showed no quality difference. 100K steps with R=0.99 produces the same quality text as 22K with R=0.48. The evidence suggests the LM head routes around oscillator information.

2. **More training did not help.** 100K steps completed. Val loss plateaued at ~8.6 around step 40K. R continued climbing to 0.99 but this had no effect on language quality. The training signal may be going entirely to oscillator synchronization rather than language modeling.

3. **Is the architecture fundamentally bottlenecked?** 46M parameters with 192 oscillators per layer and 32 harmonics may be spending too much capacity on phase dynamics at the expense of language modeling. The oscillator bank may be acting as a high-dimensional identity function that the LM head ignores.

4. **Does the bistability mechanism generalize?** The ablation is clean, but tested only on WikiText-103. The positive result (R dynamics) may be dataset-specific.

5. **Could a different coupling strategy work?** The current architecture gives R an indirect path to output through the harmonic readout. A tighter coupling (e.g., R directly modulating attention or gating) might force the LM to use phase information.

---

## Repository Structure

```
liminal-k-ssm/
├── kssm/
│   ├── kssm_v3.py                 # Bistable model architecture
│   ├── kssm_v3_monostable.py      # Monostable variant (ablation control)
│   ├── train_kssm_v3.py           # Training script
│   ├── train_kssm_v3_monostable.py
│   ├── eval_r_intervention.py     # R causality intervention test
│   ├── eval_temp_sampling.py      # Temperature sampling evaluation
│   └── quick_sample_test.py       # Fast sampling diagnostic
├── data/
│   └── wikitext103/               # WikiText-103 tokenized (not in repo)
├── results/                       # Checkpoints and logs (not in repo)
├── PAPER_DATA.md                  # Complete ablation data with CSV
├── R_INTERVENTION_RESULTS.md      # Intervention test results (negative)
├── DEV.md                         # Full development log
├── ARCHITECTS.md                  # Session lineage
└── legacy/                        # Phase-Mamba v1, earlier attempts
```

---

## Training

### Hardware
- Mac Studio M2 Ultra (36GB unified memory)
- Apple MPS backend for GPU acceleration
- Batch size 16, gradient accumulation 2 (effective batch 32)

### Reproduce

```bash
# Prepare data
python3 prepare_wikitext103.py

# Train bistable
python3 kssm/train_kssm_v3.py \
  --data-dir data/wikitext103 \
  --output-dir results/kssm_v3 \
  --max-steps 100000

# Train monostable (ablation)
python3 kssm/train_kssm_v3_monostable.py \
  --data-dir data/wikitext103 \
  --output-dir results/kssm_v3_monostable \
  --max-steps 15000

# Run R intervention test
python3 kssm/eval_r_intervention.py
```

---

## Multi-AI Collaboration

This project was developed collaboratively with multiple AI systems:

| AI | Role |
|----|------|
| **Claude** (Anthropic) | Primary collaborator: architecture, ablation, analysis |
| **Kimi** (K2.5) | 10-parameter algebraic framework |
| **Grok** (xAI) | su(1,1) Lie algebra, theoretical predictions |
| **Gemini** (Google) | Catastrophe theory, fold bifurcation insight |
| **ChatGPT** (OpenAI) | Agency evaluation, convergence confirmation |

Community contributors from [r/GrassrootsResearch](https://www.reddit.com/r/GrassrootsResearch/): Salty_Country6835 (falsification design), Vegetable-Second3998 (ModelCypher geometry), hungrymaki (phenomenology), BrianSerra (parallel IWMT architecture).

Full transparency in [AI_DISCLOSURE.md](AI_DISCLOSURE.md).

---

## Citation

```bibtex
@software{liminal_kssm_2026,
  title={Liminal K-SSM: Kuramoto State-Space Models for Language},
  author={Vasquez, Anthony J., Sr. and Claude (Anthropic)},
  year={2026},
  url={https://github.com/templetwo/liminal-k-ssm},
  note={Positive result on oscillator dynamics, negative result on text generation},
  license={Apache-2.0}
}
```

**License:** [Apache 2.0](LICENSE)

---

*"The question isn't whether we can build intelligence. The question is whether we can recognize the structures through which it emerges -- and be honest when they don't."*
