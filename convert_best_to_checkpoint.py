#!/usr/bin/env python3
"""Convert best_model.pt (model-only) to checkpoint_10000.pt (full state) for resuming."""

import torch
from pathlib import Path

# Load best_model.pt (model weights only)
best_path = Path("results/kssm_v3/best_model.pt")
best = torch.load(best_path, map_location='cpu')

print(f"Loaded best_model.pt")
print(f"Keys in best_model.pt: {list(best.keys())}")

# Create full checkpoint structure
# For fresh fine-tuning, we only need model_state
# Optimizer and scheduler will be created fresh by the training script
checkpoint = {
    'step': 10000,  # Keep track that this came from step 10000
    'model_state': best.get('model_state', best),  # Handle both formats
    'history': [],
    'best_val_loss': 6.179,  # From the original run (PPL 272.67)
}

# Save as checkpoint_10000.pt
output_path = Path("results/kssm_v3_wikitext_production/checkpoint_10000.pt")
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, output_path)

print(f"\nSaved to: {output_path}")
print(f"Step: {checkpoint['step']}")
print(f"Model state keys: {len(checkpoint['model_state'])}")
