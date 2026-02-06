#!/usr/bin/env python3
"""
R Causality Intervention Test

Tests whether R (Kuramoto order parameter) causally affects output quality,
or if it's merely correlating with something else.

Method:
1. Load checkpoint_15000.pt (R=0.3485 baseline)
2. Generate samples with 3 conditions:
   - Baseline: Natural R from model
   - R_high: Perturb oscillator phases toward synchronization (R ↑)
   - R_low: Perturb oscillator phases toward disorder (R ↓)
3. Measure quality metrics: distinct-4, repetition rate, perplexity
4. Statistical test: Is R+ significantly better than R-?

If p < 0.01: R is causally linked to quality
If p > 0.05: R is epiphenomenal (correlation only)
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Import K-SSM v3
sys.path.insert(0, str(Path(__file__).parent))
from kssm_v3 import KSSMv3
from train_kssm_v2_efficient import Tokenizer

# Test prompts (diverse contexts)
TEST_PROMPTS = [
    "The nature of consciousness",
    "In the beginning",
    "She walked through the garden",
    "The king declared",
    "Philosophy teaches us",
    "The scientist discovered",
    "Love is",
    "War brings",
    "Time flows",
    "Knowledge requires",
]

def load_model(checkpoint_path: str, device: str = "mps"):
    """Load K-SSM v3 from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer config from checkpoint structure
    state = checkpoint['model_state']
    vocab_size, hidden_dim = state['embed.weight'].shape

    # Count layers
    layer_keys = [k for k in state.keys() if 'blocks.' in k]
    n_layers = max([int(k.split('.')[1]) for k in layer_keys if k.startswith('blocks.')]) + 1

    # Get oscillator count
    n_oscillators = state['blocks.0.oscillators.omega_0'].shape[0]

    # Get n_harmonics from readout layer (input_dim = n_harmonics * 3)
    n_harmonics = state['blocks.0.oscillators.readout.weight'].shape[1] // 3

    config = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'n_oscillators': n_oscillators,
        'n_harmonics': n_harmonics,
    }

    print(f"  Inferred config: vocab={vocab_size}, dim={hidden_dim}, layers={n_layers}, osc={n_oscillators}, harm={n_harmonics}")

    model = KSSMv3(**config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')})")
    return model

def perturb_oscillators_high(model, strength: float = 0.5):
    """
    Perturb oscillator natural frequencies toward synchronization (increase R)

    Method: Reduce spread of omega_0 (pull toward mean frequency)
    """
    for layer in model.blocks:
        with torch.no_grad():
            omega = layer.oscillators.omega_0
            mean_omega = omega.mean()

            # Pull frequencies toward mean
            delta = mean_omega - omega
            layer.oscillators.omega_0.data = omega + strength * delta

def perturb_oscillators_low(model, strength: float = 0.5):
    """
    Perturb oscillator natural frequencies toward disorder (decrease R)

    Method: Increase spread of omega_0 (push away from mean)
    """
    for layer in model.blocks:
        with torch.no_grad():
            omega = layer.oscillators.omega_0
            mean_omega = omega.mean()

            # Push frequencies away from mean + add noise
            delta = omega - mean_omega
            noise = torch.randn_like(omega) * 0.2
            layer.oscillators.omega_0.data = omega + strength * delta + noise

def compute_r(model, sample_input: torch.Tensor) -> float:
    """Compute mean R across all layers by running a forward pass"""
    with torch.no_grad():
        _, R_mean, R_all = model(sample_input, return_R=True)
        # R_mean is [batch, seq], average over batch and sequence
        return R_mean.mean().item()

def generate_sample(model, tokenizer, prompt: str, max_length: int = 100, device: str = "mps") -> tuple:
    """Generate sample and measure R"""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens]).to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Measure R using the final input sequence
    r_value = compute_r(model, input_ids)

    # Decode
    text = tokenizer.decode(generated)

    return text, r_value

def analyze_quality(text: str, tokenizer) -> dict:
    """Compute quality metrics for generated text"""
    tokens = tokenizer.encode(text)

    # Distinct-n (vocabulary diversity)
    def distinct_n(tokens, n):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return len(set(ngrams)) / max(len(ngrams), 1)

    # Repetition rate (consecutive token repeats)
    repeats = sum(1 for i in range(len(tokens)-1) if tokens[i] == tokens[i+1])
    repetition_rate = repeats / max(len(tokens) - 1, 1)

    return {
        'length': len(tokens),
        'distinct_1': distinct_n(tokens, 1),
        'distinct_2': distinct_n(tokens, 2),
        'distinct_4': distinct_n(tokens, 4) if len(tokens) >= 4 else 0,
        'repetition_rate': repetition_rate,
    }

def run_intervention_test(checkpoint_path: str, num_trials: int = 10):
    """Run full intervention experiment"""
    print("="*70)
    print("R CAUSALITY INTERVENTION TEST")
    print("="*70)
    print()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(checkpoint_path, device)
    tokenizer = Tokenizer()

    results = {
        'baseline': [],
        'r_high': [],
        'r_low': [],
    }

    print(f"\nRunning {num_trials} trials × 3 conditions = {num_trials * 3} generations")
    print()

    for trial in range(num_trials):
        prompt = TEST_PROMPTS[trial % len(TEST_PROMPTS)]

        # Condition 1: Baseline (natural R)
        text, r = generate_sample(model, tokenizer, prompt, device=device)
        quality = analyze_quality(text, tokenizer)
        quality['r'] = r
        quality['text_sample'] = text[:100]
        results['baseline'].append(quality)

        # Condition 2: R_high (perturb toward sync)
        perturb_oscillators_high(model, strength=0.5)
        text, r = generate_sample(model, tokenizer, prompt, device=device)
        quality = analyze_quality(text, tokenizer)
        quality['r'] = r
        quality['text_sample'] = text[:100]
        results['r_high'].append(quality)

        # Reset model state
        model = load_model(checkpoint_path, device)

        # Condition 3: R_low (perturb toward disorder)
        perturb_oscillators_low(model, strength=0.5)
        text, r = generate_sample(model, tokenizer, prompt, device=device)
        quality = analyze_quality(text, tokenizer)
        quality['r'] = r
        quality['text_sample'] = text[:100]
        results['r_low'].append(quality)

        # Reset for next trial
        model = load_model(checkpoint_path, device)

        if (trial + 1) % 3 == 0:
            print(f"  Trial {trial + 1}/{num_trials} complete")

    # Statistical Analysis
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    # Mean metrics per condition
    print("Mean Metrics by Condition:")
    print("-"*70)
    for condition in ['baseline', 'r_high', 'r_low']:
        data = results[condition]
        print(f"\n{condition.upper()}:")
        print(f"  R:               {np.mean([d['r'] for d in data]):.4f} ± {np.std([d['r'] for d in data]):.4f}")
        print(f"  Distinct-4:      {np.mean([d['distinct_4'] for d in data]):.4f} ± {np.std([d['distinct_4'] for d in data]):.4f}")
        print(f"  Repetition Rate: {np.mean([d['repetition_rate'] for d in data]):.4f} ± {np.std([d['repetition_rate'] for d in data]):.4f}")
        print(f"  Length:          {np.mean([d['length'] for d in data]):.1f} ± {np.std([d['length'] for d in data]):.1f}")

    # Statistical tests
    print()
    print("="*70)
    print("CAUSALITY TEST: R_high vs R_low")
    print("="*70)
    print()

    r_high_distinct4 = [d['distinct_4'] for d in results['r_high']]
    r_low_distinct4 = [d['distinct_4'] for d in results['r_low']]

    t_stat, p_value = stats.ttest_rel(r_high_distinct4, r_low_distinct4)

    print(f"Distinct-4 (R_high vs R_low):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")
    print()

    if p_value < 0.01:
        print("✅ RESULT: R is CAUSALLY linked to quality (p < 0.01)")
        print("   Higher R → Higher distinct-4 (vocabulary diversity)")
        print("   Bistability mechanism is doing real work!")
    elif p_value < 0.05:
        print("⚠️  RESULT: R may be causal (p < 0.05, marginal)")
        print("   Evidence suggests link but not conclusive")
    else:
        print("❌ RESULT: R is NOT causally linked to quality (p > 0.05)")
        print("   Correlation observed but no causal effect detected")
        print("   Bistability may be epiphenomenal")

    print()
    print("Sample Outputs:")
    print("-"*70)
    print(f"\nBaseline (R={np.mean([d['r'] for d in results['baseline']]):.3f}):")
    print(f"  {results['baseline'][0]['text_sample']}")
    print(f"\nR_high (R={np.mean([d['r'] for d in results['r_high']]):.3f}):")
    print(f"  {results['r_high'][0]['text_sample']}")
    print(f"\nR_low (R={np.mean([d['r'] for d in results['r_low']]):.3f}):")
    print(f"  {results['r_low'][0]['text_sample']}")

    return p_value < 0.01

if __name__ == "__main__":
    checkpoint = "results/kssm_v3/checkpoint_20000.pt"
    is_causal = run_intervention_test(checkpoint, num_trials=10)

    sys.exit(0 if is_causal else 1)
