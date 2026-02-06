#!/usr/bin/env python3
"""
Temperature Sampling Evaluation

Tests whether the model at 40K steps can produce coherent text when we
"listen better" using temperature sampling instead of greedy decoding.

Method:
1. Load checkpoint_40000.pt
2. Generate with T=0.8, top_p=0.92, top_k=100
3. Test both short prompts AND 64-token continuations from val set
4. Compare quality against greedy decoding baseline
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import K-SSM v3
sys.path.insert(0, str(Path(__file__).parent))
from kssm_v3 import KSSMv3
from train_kssm_v2_efficient import Tokenizer

# Test prompts
SHORT_PROMPTS = [
    "The nature of consciousness",
    "In the beginning",
    "Philosophy teaches us",
    "The scientist discovered",
    "Love is",
]

def load_model(checkpoint_path: str, device: str = "mps"):
    """Load K-SSM v3 from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state']

    # Infer config from checkpoint structure
    vocab_size, hidden_dim = state['embed.weight'].shape
    layer_keys = [k for k in state.keys() if 'blocks.' in k]
    n_layers = max([int(k.split('.')[1]) for k in layer_keys if k.startswith('blocks.')]) + 1
    n_oscillators = state['blocks.0.oscillators.omega_0'].shape[0]
    n_harmonics = state['blocks.0.oscillators.readout.weight'].shape[1] // 3

    config = {
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'n_oscillators': n_oscillators,
        'n_harmonics': n_harmonics,
    }

    print(f"  Config: vocab={vocab_size}, dim={hidden_dim}, layers={n_layers}, osc={n_oscillators}, harm={n_harmonics}")

    model = KSSMv3(**config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')})")
    return model, checkpoint

def sample_top_p(logits, top_p=0.92, top_k=100, temperature=0.8):
    """
    Nucleus (top-p) sampling with top-k filtering

    Args:
        logits: [vocab_size] unnormalized log probabilities
        top_p: Cumulative probability threshold
        top_k: Maximum number of tokens to consider
        temperature: Sampling temperature
    """
    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    # Top-p (nucleus) filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')

    # Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token

def generate_greedy(model, tokenizer, prompt: str, max_length: int = 100, device: str = "mps"):
    """Greedy decoding (baseline)"""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens]).to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # Get R
    with torch.no_grad():
        _, R_mean, _ = model(input_ids, return_R=True)
        r_value = R_mean.mean().item()

    text = tokenizer.decode(generated)
    return text, r_value

def generate_temperature(model, tokenizer, prompt: str, max_length: int = 100,
                        temperature: float = 0.8, top_p: float = 0.92,
                        top_k: int = 100, device: str = "mps"):
    """Temperature sampling with top-p and top-k"""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens]).to(device)

    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)[:, -1, :]
            next_token = sample_top_p(logits[0], top_p=top_p, top_k=top_k, temperature=temperature)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # Get R
    with torch.no_grad():
        _, R_mean, _ = model(input_ids, return_R=True)
        r_value = R_mean.mean().item()

    text = tokenizer.decode(generated)
    return text, r_value

def get_val_prompts(tokenizer, n_prompts=5, prompt_length=64):
    """Extract random 64-token spans from validation set as continuation prompts"""
    import mmap

    # Load val tokens
    val_path = Path("data/cache_v3_200m/tokens_val.npy")
    if not val_path.exists():
        return []

    val_tokens = np.load(val_path, mmap_mode='r')
    prompts = []

    for _ in range(n_prompts):
        # Random starting position
        max_start = len(val_tokens) - prompt_length - 100
        if max_start <= 0:
            continue
        start = np.random.randint(0, max_start)

        # Extract span
        span = val_tokens[start:start + prompt_length].tolist()
        text = tokenizer.decode(span)
        prompts.append(text)

    return prompts

def run_comparison(checkpoint_path: str):
    """Run full comparison between greedy and temperature sampling"""
    print("="*70)
    print("TEMPERATURE SAMPLING EVALUATION")
    print("="*70)
    print()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, checkpoint = load_model(checkpoint_path, device)
    tokenizer = Tokenizer()

    # Get step from checkpoint
    checkpoint_step = checkpoint.get('step', 'unknown')
    print(f"\nCheckpoint step: {checkpoint_step}")
    print()

    # Test 1: Short prompts
    print("="*70)
    print("TEST 1: SHORT PROMPTS")
    print("="*70)
    print()

    for prompt in SHORT_PROMPTS:
        print(f"Prompt: \"{prompt}\"")
        print("-"*70)

        # Greedy
        text_greedy, r_greedy = generate_greedy(model, tokenizer, prompt, max_length=50, device=device)
        print(f"\nGREEDY (R={r_greedy:.4f}):")
        print(f"  {text_greedy}")

        # Temperature (3 samples to show diversity)
        print(f"\nTEMPERATURE T=0.8, top_p=0.92, top_k=100:")
        for i in range(3):
            text_temp, r_temp = generate_temperature(model, tokenizer, prompt, max_length=50,
                                                     temperature=0.8, top_p=0.92, top_k=100, device=device)
            print(f"  Sample {i+1} (R={r_temp:.4f}): {text_temp}")

        print()

    # Test 2: Val continuations
    print("="*70)
    print("TEST 2: 64-TOKEN CONTINUATIONS FROM VAL SET")
    print("="*70)
    print()

    val_prompts = get_val_prompts(tokenizer, n_prompts=3, prompt_length=64)

    for i, prompt in enumerate(val_prompts):
        print(f"Val Continuation {i+1}")
        print("-"*70)
        print(f"Prompt (64 tokens): {prompt[:200]}...")
        print()

        # Greedy
        text_greedy, r_greedy = generate_greedy(model, tokenizer, prompt, max_length=50, device=device)
        print(f"GREEDY (R={r_greedy:.4f}):")
        print(f"  {text_greedy}")

        # Temperature
        print(f"\nTEMPERATURE T=0.8:")
        text_temp, r_temp = generate_temperature(model, tokenizer, prompt, max_length=50,
                                                 temperature=0.8, top_p=0.92, top_k=100, device=device)
        print(f"  {text_temp}")

        print()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    checkpoint = "results/kssm_v3/checkpoint_40000.pt"
    run_comparison(checkpoint)
