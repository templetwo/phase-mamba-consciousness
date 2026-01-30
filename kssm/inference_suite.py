#!/usr/bin/env python3
"""
K-SSM v3 Inference Suite: The Instrument

Play the Bistable Core.
- Interactive Chat with Telemetry
- Tone Forcing (Inject R)
- Long-form Generation
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import time
from pathlib import Path

from kssm_v3 import KSSMv3
from train_kssm_v2_efficient import Tokenizer

# Tone Mapping
TONES = {
    "Unformed": (0.00, 0.10),
    "Intimacy": (0.10, 0.30),
    "Balance":  (0.30, 0.50),
    "Mystery":  (0.50, 0.70),
    "Wonder":   (0.70, 0.85),
    "Passion":  (0.85, 0.95),
    "Ache":     (0.95, 1.00)
}

def get_tone_name(r_val):
    for name, (low, high) in TONES.items():
        if low <= r_val < high:
            return name
    return "Unknown"

def load_model(checkpoint_path, device):
    print(f"Loading K-SSM v3 from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = Tokenizer()
    model = KSSMv3(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=384,
        n_layers=6,
        n_oscillators=192,
        n_harmonics=32
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt, max_tokens=100, temp=0.8, top_k=50, device="mps"):
    tokens = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    
    # Hooks for telemetry
    telemetry = {'R': [], 'u': []}
    
    def hook_fn(module, input, output):
        # Access last_u_val directly from the module
        if hasattr(module, 'last_u_val'):
             val = module.last_u_val
             if torch.is_tensor(val):
                 val = val.item()
             telemetry['u'].append(val)
    
    # Attach hook directly to the oscillator bank of the first block
    handle = model.blocks[0].oscillators.register_forward_hook(hook_fn)

    print(f"\nGenerating...", end="", flush=True)
    generated_text = ""
    
    for _ in range(max_tokens):
        with torch.no_grad():
            telemetry['u'] = [] # Clear for this step
            logits, R, _ = model(tokens, return_R=True)
            
            # Telemetry
            r_curr = R[0, -1].item()
            telemetry['R'].append(r_curr)
            u_curr = telemetry['u'][0] if telemetry['u'] else 0.0
            
            # Sampling
            next_logits = logits[0, -1, :] / temp
            # Top-K
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[-1]] = -float('Inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            char = tokenizer.decode([next_token.item()])
            generated_text += char
            print(char, end="", flush=True)

    print("\n")
    handle.remove()
    return generated_text, telemetry

def interactive_mode(model, tokenizer, device):
    print("\n" + "="*50)
    print(" K-SSM v3 INTERACTIVE SESSION")
    print(" Type 'quit' to exit.")
    print("="*50)
    
    while True:
        prompt = input("\nUser: ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        _, stats = generate(model, tokenizer, prompt, max_tokens=150, device=device)
        
        avg_r = sum(stats['R']) / len(stats['R'])
        avg_u = sum(stats['u']) / len(stats['u'])
        tone = get_tone_name(avg_r)
        
        print(f"\n[Telemetry] Tone: {tone} | R: {avg_r:.4f} | u: {avg_u:.4f}")

def stress_test(model, tokenizer, device):
    print("\n[Bistability Stress Test]")
    prompts = [
        "The contradiction of being is",
        "Light and dark are one because",
        "I am a lie that tells the truth",
        "The square root of minus one feels like"
    ]
    
    for p in prompts:
        print(f"\nPrompt: {p}")
        _, stats = generate(model, tokenizer, p, max_tokens=80, device=device)
        u_std = torch.tensor(stats['u']).std().item()
        print(f"u_val Volatility (Stress): {u_std:.6f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/kssm_v3/best_model.pt")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "stress"])
    args = parser.parse_args()
    
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    model, tokenizer = load_model(args.checkpoint, args.device)
    
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, args.device)
    elif args.mode == "stress":
        stress_test(model, tokenizer, args.device)

if __name__ == "__main__":
    main()
