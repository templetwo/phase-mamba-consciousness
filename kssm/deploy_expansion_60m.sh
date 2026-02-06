#!/bin/bash

# K-SSM Corpus Expansion: 41M → 60M Tokens
# Adds 95 new English books to existing corpus

set -e

REMOTE_USER="tony_studio"
REMOTE_HOST="192.168.1.195"
REMOTE_DIR="~/liminal-k-ssm"
REMOTE_ADDR="${REMOTE_USER}@${REMOTE_HOST}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "======================================================================"
echo "K-SSM CORPUS EXPANSION: 41M → 60M TOKENS"
echo "======================================================================"
echo ""
echo "Adding 95 new English books to existing 206 books"
echo "Estimated result: ~301 books, ~60M tokens"
echo ""
echo "Target: ${REMOTE_ADDR}:${REMOTE_DIR}"
echo ""

# Step 1: Transfer expansion script
echo "${YELLOW}[1/4] TRANSFERRING EXPANSION SCRIPT${NC}"
echo "----------------------------------------------------------------------"

echo "  Uploading corpus_expansion_to_60m.py..."
scp kssm/corpus_expansion_to_60m.py ${REMOTE_ADDR}:${REMOTE_DIR}/kssm/

echo "  ✓ Script transferred"
echo ""

# Step 2: Download new books
echo "${YELLOW}[2/4] DOWNLOADING 95 NEW BOOKS${NC}"
echo "----------------------------------------------------------------------"

ssh ${REMOTE_ADDR} << 'ENDSSH'
cd ~/liminal-k-ssm

echo "  Starting download..."
echo "  Estimated time: ~4 minutes (95 books × 2.5s rate limit)"
echo ""

python3 << 'ENDPYTHON'
import sys
import time
sys.path.insert(0, 'kssm')

from corpus_expansion_to_60m import NEW_ENGLISH_BOOKS
from build_corpus_200m import download_gutenberg_book

print(f"Downloading {len(NEW_ENGLISH_BOOKS)} new books...")

success = 0
failed = []

for i, (book_id, title, author) in enumerate(NEW_ENGLISH_BOOKS, 1):
    print(f"  [{i}/{len(NEW_ENGLISH_BOOKS)}] {title} - {author}")
    if download_gutenberg_book(book_id, title, author):
        success += 1
    else:
        failed.append((book_id, title, author))
    time.sleep(2)  # Rate limiting

print(f"\n  ✓ Downloaded: {success}/{len(NEW_ENGLISH_BOOKS)}")
if failed:
    print(f"  ✗ Failed: {len(failed)}")
    for book_id, title, _ in failed[:5]:
        print(f"    - {book_id}: {title}")

ENDPYTHON

ENDSSH

echo ""

# Step 3: Rebuild corpus with all books
echo "${YELLOW}[3/4] REBUILDING CORPUS JSONL${NC}"
echo "----------------------------------------------------------------------"

ssh ${REMOTE_ADDR} << 'ENDSSH'
cd ~/liminal-k-ssm

echo "  Creating updated build script..."

# Create updated build_corpus_200m.py that uses merged book list
python3 << 'ENDPYTHON'
import sys
sys.path.insert(0, 'kssm')
from corpus_expansion_to_60m import ALL_BOOKS

print(f"  Processing {len(ALL_BOOKS)} books...")

# Import and run build logic
from build_corpus_200m import RAW_DIR, PROCESSED_DIR, build_corpus, clean_text, chunk_text
import os
import json
from tqdm import tqdm
from pathlib import Path

# Build corpus
all_chunks = []
book_stats = []

for book_id, title, author in tqdm(ALL_BOOKS, desc="Processing books"):
    # Find downloaded file
    filename = f"{book_id}_{author.replace(' ', '_')}_{title.replace(' ', '_')}.txt"
    filepath = os.path.join(RAW_DIR, "gutenberg", filename)

    if not os.path.exists(filepath):
        continue

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    for chunk in chunks:
        all_chunks.append({
            "text": chunk,
            "source": "gutenberg",
            "book_id": book_id,
            "title": title,
            "author": author
        })

    book_stats.append({
        "book_id": book_id,
        "title": title,
        "author": author,
        "chunks": len(chunks)
    })

# Write JSONL
output_path = os.path.join(PROCESSED_DIR, "kssm_corpus_60m.jsonl")
with open(output_path, 'w') as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + '\n')

print(f"\n  ✓ Created corpus: {len(all_chunks):,} chunks")
print(f"  ✓ From {len(book_stats)} books")
print(f"  ✓ Saved to: kssm/data/processed/kssm_corpus_60m.jsonl")

# Show stats
total_text = sum(len(chunk['text']) for chunk in all_chunks)
print(f"  ✓ Total characters: {total_text:,}")
print(f"  ✓ Estimated tokens: ~{total_text / 4:,.0f}")

ENDPYTHON

ENDSSH

echo ""

# Step 4: Tokenize to numpy
echo "${YELLOW}[4/4] TOKENIZING TO NUMPY${NC}"
echo "----------------------------------------------------------------------"

ssh ${REMOTE_ADDR} << 'ENDSSH'
cd ~/liminal-k-ssm

python3 kssm/process_corpus_200m.py \
    --tokenize \
    --input kssm/data/processed/kssm_corpus_60m.jsonl \
    --output data/cache_v3_60m

echo ""
echo "  ✓ Tokenization complete"

# Show final stats
python3 << 'ENDPYTHON'
import numpy as np
import os

train_file = "data/cache_v3_60m/tokens_train.npy"
val_file = "data/cache_v3_60m/tokens_val.npy"

if os.path.exists(train_file) and os.path.exists(val_file):
    train = np.load(train_file, mmap_mode='r')
    val = np.load(val_file, mmap_mode='r')

    total = len(train) + len(val)

    print(f"\n  ✓ Train: {len(train):,} tokens")
    print(f"  ✓ Val: {len(val):,} tokens")
    print(f"  ✓ Total: {total:,} tokens ({total/1e6:.1f}M)")

    # Show file sizes
    train_size = os.path.getsize(train_file) / (1024**2)
    val_size = os.path.getsize(val_file) / (1024**2)
    print(f"\n  Train file: {train_size:.1f} MB")
    print(f"  Val file: {val_size:.1f} MB")

ENDPYTHON

ENDSSH

echo ""
echo "${GREEN}======================================================================"
echo "EXPANSION COMPLETE"
echo "======================================================================${NC}"
echo ""
echo "Corpus expanded from 41M to ~60M tokens"
echo ""
echo "New files:"
echo "  - kssm/data/processed/kssm_corpus_60m.jsonl"
echo "  - data/cache_v3_60m/tokens_train.npy"
echo "  - data/cache_v3_60m/tokens_val.npy"
echo ""
echo "Original backups preserved:"
echo "  - kssm/data/processed/kssm_corpus_22M_backup.jsonl"
echo "  - data/cache_v3_22M_backup/"
echo ""
echo "To continue to 100M or 200M tokens, curate more English books."
echo ""
