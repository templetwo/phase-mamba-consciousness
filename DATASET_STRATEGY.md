# K-SSM v3: Dataset Strategy — Standard Benchmarks

> **Decision Date**: 2026-02-01
> **Context**: Custom Gutenberg corpus found to have ~3.2% tokenization corruption.
> **Strategic Pivot**: Use well-established benchmark datasets to eliminate data quality as a variable and enable direct comparison against published baselines.

---

## Why Standard Benchmarks?

1. **Validated by thousands of researchers** — no encoding surprises
2. **Published baselines exist** — direct PPL comparison vs Mamba, Transformer, RWKV, etc.
3. **Reviewers trust them** — no questions about data quality in peer review
4. **Reproducible** — anyone can verify your results
5. **Saves weeks of corpus curation**

---

## Recommended Datasets

### Dataset 1: WikiText-103 (PRIMARY — for PPL benchmark)

| Property | Value |
|----------|-------|
| Tokens | ~103M |
| Source | Wikipedia (Good/Featured articles) |
| Vocab | Originally word-level; we retokenize with cl100k_base |
| Standard for | Language model perplexity comparison |
| Published baselines | Transformer-XL, GPT-2, Mamba, RWKV, etc. |

**Why this one**: Every language modeling paper reports WikiText-103 PPL. If K-SSM v3 (46M params) achieves PPL within 10% of vanilla Mamba (46M params) on WikiText-103, that's publishable proof that oscillator dynamics don't hurt and may help.

**Download and prepare:**
```python
from datasets import load_dataset

# Download
ds = load_dataset("wikitext", "wikitext-103-raw-v1")

# Access splits
train_text = "\n".join([x["text"] for x in ds["train"] if x["text"].strip()])
val_text = "\n".join([x["text"] for x in ds["validation"] if x["text"].strip()])
test_text = "\n".join([x["text"] for x in ds["test"] if x["text"].strip()])
```

### Dataset 2: TinyStories (SECONDARY — for coherence demo)

| Property | Value |
|----------|-------|
| Tokens | ~470M (can subset) |
| Source | GPT-3.5/4 generated short stories |
| Designed for | Training small models (1M–33M params) to generate coherent text |
| Key paper | "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (Eldan & Li, 2023) |

**Why this one**: Specifically designed to test whether small models can generate coherent, grammatical English. If K-SSM v3 generates more coherent stories than a vanilla model of the same size, that's the phase-coupling proof.

**Download and prepare:**
```python
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")
train_text = "\n".join([x["text"] for x in ds["train"]])
val_text = "\n".join([x["text"] for x in ds["validation"]])
```

---

## Implementation: Tokenization Pipeline with Validation

```python
"""
prepare_benchmark_dataset.py

Tokenizes a standard benchmark dataset with round-trip validation.
Produces .npy files compatible with K-SSM v3 training pipeline.
"""
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path

def validate_roundtrip(tokenizer, text, chunk_size=10000):
    """Ensure decode(encode(text)) == text for every chunk."""
    tokens = tokenizer.encode(text)
    errors = 0
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        decoded = tokenizer.decode(chunk)
        reencoded = tokenizer.encode(decoded)
        if chunk != reencoded:
            errors += 1
    
    # Also check for U+FFFD
    full_decoded = tokenizer.decode(tokens)
    fffd_count = full_decoded.count('\ufffd')
    
    return {
        'total_chunks': len(tokens) // chunk_size + 1,
        'roundtrip_errors': errors,
        'fffd_count': fffd_count,
        'clean': errors == 0 and fffd_count == 0
    }

def prepare_wikitext103(output_dir="data/wikitext103"):
    """Download, tokenize, validate, and save WikiText-103."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    for split in ["train", "validation", "test"]:
        print(f"\nProcessing {split}...")
        text = "\n".join([x["text"] for x in ds[split] if x["text"].strip()])
        
        # Tokenize
        tokens = enc.encode(text)
        print(f"  {len(tokens):,} tokens")
        
        # Validate
        print(f"  Validating round-trip...")
        result = validate_roundtrip(enc, text)
        print(f"  Round-trip errors: {result['roundtrip_errors']}")
        print(f"  U+FFFD count: {result['fffd_count']}")
        
        if not result['clean']:
            print(f"  ⚠️  VALIDATION FAILED — cleaning required")
            # Clean: remove lines with problematic characters
            clean_lines = []
            for line in text.split("\n"):
                line_tokens = enc.encode(line)
                decoded = enc.decode(line_tokens)
                if '\ufffd' not in decoded:
                    clean_lines.append(line)
            text = "\n".join(clean_lines)
            tokens = enc.encode(text)
            print(f"  After cleaning: {len(tokens):,} tokens")
        else:
            print(f"  ✅ CLEAN")
        
        # Save
        tokens_np = np.array(tokens, dtype=np.int32)
        outpath = output_dir / f"tokens_{split}.npy"
        np.save(outpath, tokens_np)
        print(f"  Saved to {outpath}")

def prepare_tinystories(output_dir="data/tinystories", max_tokens=60_000_000):
    """Download, tokenize, validate, and save TinyStories (subsetted)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    for split in ["train", "validation"]:
        print(f"\nProcessing {split}...")
        texts = [x["text"] for x in ds[split] if x["text"].strip()]
        
        # Tokenize incrementally with cap
        all_tokens = []
        for text in texts:
            tokens = enc.encode(text)
            all_tokens.extend(tokens)
            if split == "train" and len(all_tokens) >= max_tokens:
                all_tokens = all_tokens[:max_tokens]
                break
        
        print(f"  {len(all_tokens):,} tokens")
        
        # Validate sample
        sample_decoded = enc.decode(all_tokens[:50000])
        fffd = sample_decoded.count('\ufffd')
        print(f"  U+FFFD in 50K sample: {fffd}")
        if fffd == 0:
            print(f"  ✅ CLEAN")
        
        # Save
        tokens_np = np.array(all_tokens, dtype=np.int32)
        outpath = output_dir / f"tokens_{split}.npy"
        np.save(outpath, tokens_np)
        print(f"  Saved to {outpath}")

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "wikitext103"
    
    if dataset == "wikitext103":
        prepare_wikitext103()
    elif dataset == "tinystories":
        prepare_tinystories()
    elif dataset == "both":
        prepare_wikitext103()
        prepare_tinystories()
    else:
        print(f"Unknown dataset: {dataset}")
        print("Usage: python prepare_benchmark_dataset.py [wikitext103|tinystories|both]")
```

---

## Comparison Baselines to Beat

For a ~46M parameter model on WikiText-103:

| Model | Params | WikiText-103 PPL | Source |
|-------|--------|------------------|--------|
| Transformer (small) | 44M | ~35-40 | Various |
| Mamba (small) | 45M | ~30-35 | Gu & Dao 2023 |
| RWKV (small) | 44M | ~35-38 | Peng et al 2023 |
| **K-SSM v3** | 46M | **TBD** | This work |

**Target**: PPL within 20% of vanilla Mamba at same scale would be strong. Within 10% or better would be exceptional.

**The real finding**: Even if PPL is slightly worse, if R intervention shows causal coherence effect, that's a novel contribution no other architecture has.

---

## Training Plan on Clean Data

```
Phase 1: WikiText-103 training
- Fresh K-SSM v3 from random init
- Same hyperparams as current run
- Train for 40K+ steps
- Log R, PPL, u_val, Δ (new!) every 500 steps
- Run R intervention test at 10K, 20K, 30K, 40K

Phase 2: TinyStories training (parallel or sequential)
- Same architecture
- Train until coherent story generation
- Compare output quality vs vanilla Mamba baseline

Phase 3: Ablation studies
- Harmonic reduction: [8, 16, 32] oscillators
- Coupling strength: vary K
- Bistability: vary u_min

Phase 4: Paper
- WikiText-103 PPL comparison table
- R causality intervention results
- TinyStories qualitative comparison
- Mathematical framework (Kimi + Grok)
- Antifragility / stochastic resonance finding
```

---

## Dependencies

```bash
pip install datasets tiktoken numpy --break-system-packages
```

Note: `datasets` requires internet access to download from HuggingFace Hub.
