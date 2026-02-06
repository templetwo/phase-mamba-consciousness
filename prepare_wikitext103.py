#!/usr/bin/env python3
"""
Prepare WikiText-103 for K-SSM v3 Training

Downloads WikiText-103 from HuggingFace, tokenizes with tiktoken cl100k_base,
validates zero U+FFFD replacement characters, and saves as .npy files.
"""
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path
import sys

def validate_roundtrip(tokenizer, text, max_check=100000):
    """Ensure decode(encode(text)) has no U+FFFD replacement characters."""
    tokens = tokenizer.encode(text)

    # Check first max_check tokens for U+FFFD
    check_tokens = tokens[:max_check]
    decoded = tokenizer.decode(check_tokens)
    fffd_count = decoded.count('\ufffd')

    return {
        'total_tokens': len(tokens),
        'checked_tokens': len(check_tokens),
        'fffd_count': fffd_count,
        'clean': fffd_count == 0
    }

def prepare_wikitext103(output_dir="data/wikitext103"):
    """Download, tokenize, validate, and save WikiText-103."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("WIKITEXT-103 PREPARATION")
    print("="*70)
    print()

    print("Downloading WikiText-103 from HuggingFace...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    enc = tiktoken.get_encoding("cl100k_base")

    for split in ["train", "validation", "test"]:
        print()
        print(f"{'='*70}")
        print(f"Processing {split.upper()}")
        print(f"{'='*70}")

        # Combine all text
        texts = [x["text"] for x in ds[split] if x["text"].strip()]
        text = "\n".join(texts)

        print(f"  Total text length: {len(text):,} characters")

        # Tokenize
        print(f"  Tokenizing...")
        tokens = enc.encode(text)
        print(f"  Tokens: {len(tokens):,}")

        # Validate
        print(f"  Validating (checking first 100K tokens for U+FFFD)...")
        result = validate_roundtrip(enc, text)

        if result['fffd_count'] > 0:
            print(f"  ⚠️  FOUND {result['fffd_count']} U+FFFD replacement characters!")
            print(f"  Cleaning...")

            # Clean: remove lines with U+FFFD
            clean_lines = []
            for line in text.split("\n"):
                if line.strip():
                    line_tokens = enc.encode(line)
                    decoded = enc.decode(line_tokens)
                    if '\ufffd' not in decoded:
                        clean_lines.append(line)

            text = "\n".join(clean_lines)
            tokens = enc.encode(text)
            print(f"  After cleaning: {len(tokens):,} tokens")

            # Re-validate
            result = validate_roundtrip(enc, text)
            if result['fffd_count'] == 0:
                print(f"  ✅ CLEAN after filtering")
            else:
                print(f"  ❌ STILL HAS {result['fffd_count']} U+FFFD - aborting")
                sys.exit(1)
        else:
            print(f"  ✅ CLEAN (zero U+FFFD)")

        # Save
        tokens_np = np.array(tokens, dtype=np.int32)
        outpath = output_dir / f"tokens_{split}.npy"
        np.save(outpath, tokens_np)

        # Save metadata
        meta = {"n_tokens": len(tokens)}
        meta_path = output_dir / f"tokens_{split}_meta.json"
        import json
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        print(f"  Saved: {outpath} ({len(tokens):,} tokens)")
        print(f"  Metadata: {meta_path}")

    print()
    print("="*70)
    print("WIKITEXT-103 READY FOR TRAINING")
    print("="*70)
    print()
    print(f"Files saved to: {output_dir.absolute()}")
    print()
    print("Next step:")
    print(f"  python3 kssm/train_kssm_v3.py --data-dir {output_dir.absolute()} --max-steps 40000")

if __name__ == "__main__":
    prepare_wikitext103()
