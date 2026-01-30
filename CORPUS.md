# K-SSM v3 Training Corpus Documentation

**Complete transparency on training data sources, licensing, and processing.**

---

## Summary

**Total Tokens**: 22,233,157
- Training: 21,169,038 tokens (95.2%)
- Validation: 1,064,119 tokens (4.8%)

**Source**: 96 texts from Project Gutenberg (Public Domain)
**License**: All training data is **Public Domain** (no licensing restrictions)
**Processing**: JSONL format with metadata, tokenized via GPT-2 BPE tokenizer

---

## Data Sources

### Project Gutenberg (100% of corpus)

**License**: Public Domain (no copyright restrictions)
**Texts**: 96 books successfully downloaded and processed
**Categories**:
- Classic literature (novels, plays, poetry)
- Philosophy (Plato, Aristotle, Kant, Hume, Nietzsche, Spinoza, etc.)
- Religious texts (Bible, Quran, Bhagavad Gita, Buddhist texts)
- Historical works (Russian novels, Shakespeare, etc.)

**Download method**: Automated via `kssm/build_corpus.py`
**Source URL**: https://www.gutenberg.org/
**Download date**: January 29, 2026

**Note**: The build script (`build_corpus.py`) defines 101 Gutenberg books, but only 96 were successfully downloaded/processed. 5 books failed download or were filtered out during processing.

### OpenStax Textbooks (NOT included)

**Status**: ❌ **NOT INCLUDED** in final corpus

The build script includes definitions for 10 OpenStax textbooks (CC BY 4.0):
- Psychology 2e
- American Government 3e
- Introduction to Sociology 3e
- US History
- World History Volume 1
- Introduction to Philosophy
- Writing Guide
- Principles of Economics 3e
- Biology 2e
- Astronomy 2e

**Why not included**: OpenStax content was never downloaded. The `data/raw/openstax/` directory exists but is empty (0 files).

**Result**: Corpus is 100% Project Gutenberg (Public Domain), no CC BY 4.0 content.

---

## File Locations

### Processed Corpus

**Path**: `kssm/data/processed/kssm_corpus.jsonl`
**Size**: 93 MB
**Format**: JSONL (one document per line)
**Lines**: 21,312 chunks

**Structure**:
```json
{
  "text": "Document text content...",
  "source": "gutenberg",
  "title": "Pride and Prejudice",
  "author": "Jane Austen",
  "license": "Public Domain",
  "doc_id": "gutenberg_1342",
  "chunk_id": 0
}
```

### Raw Text Files

**Path**: `kssm/data/raw/gutenberg/`
**Files**: 96 `.txt` files
**Naming**: `{book_id}_{sanitized_title}.txt`

### Tokenized Data (Memory-Mapped)

**Training tokens**: `data/cache_v3/tokens_train.npy`
- Size: 81 MB
- Tokens: 21,169,038
- Format: NumPy array (int32)
- Metadata: `tokens_train_meta.json`

**Validation tokens**: `data/cache_v3/tokens_val.npy`
- Size: 4.1 MB
- Tokens: 1,064,119
- Format: NumPy array (int32)
- Metadata: `tokens_val_meta.json`

**Tokenizer**: GPT-2 BPE tokenizer (via tiktoken)
**Vocab size**: 50,257

---

## Processing Pipeline

### 1. Text Acquisition (`build_corpus.py`)

```python
# Download from Project Gutenberg
def download_gutenberg_book(book_id: int, title: str) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    # Downloads UTF-8 plain text version
```

### 2. Text Cleaning

- Unicode normalization (UTF-8)
- Whitespace normalization (remove excessive newlines, spaces)
- Control character removal (keep newlines)
- HTML entity decoding
- Gutenberg header/footer removal

### 3. Chunking

**Strategy**: Fixed-size chunks with overlap
- **Chunk size**: ~1024 tokens (approximate via `words * 1.3`)
- **Overlap**: 128 tokens between consecutive chunks
- **Minimum**: 100 words per chunk (prevents EOS collapse)

**Purpose**: Maintain context continuity while fitting in model sequence length (128 tokens)

### 4. Tokenization

**Method**: GPT-2 BPE tokenizer (tiktoken library)
**Process**:
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
```

**Train/Val Split**: 95% / 5%
- Training: 21.2M tokens
- Validation: 1.1M tokens

### 5. Memory Mapping

**Format**: NumPy `.npy` files with `mmap_mode='r'`
**Benefit**: Zero RAM overhead during training (data read directly from disk)
**Metadata**: JSON sidecar files with token counts

---

## Corpus Statistics

### Token Counts (Actual)

```
Training tokens:    21,169,038  (95.2%)
Validation tokens:   1,064,119  ( 4.8%)
Total tokens:       22,233,157
```

### Chunk Statistics

```
Total chunks:       21,312
Source documents:   96
Avg tokens/chunk:   ~1,043
```

### Source Breakdown

| Source | Texts | Chunks | Tokens | Percentage |
|--------|-------|--------|--------|------------|
| Project Gutenberg | 96 | 21,312 | 22,233,157 | 100% |
| OpenStax | 0 | 0 | 0 | 0% |

---

## License Compliance

### Public Domain Status

**All training data is in the Public Domain** (no copyright restrictions).

**Project Gutenberg License**:
- Pre-1928 works (automatically public domain in US)
- No attribution required
- No restrictions on use (commercial or non-commercial)
- May be freely modified, distributed, or incorporated into derivative works

**No CC BY 4.0 content** (OpenStax was not included despite being planned)

### Model License

**K-SSM v3 model weights**: Apache 2.0 (with Research Ethics Addendum)

Since all training data is Public Domain, there are no licensing conflicts or attribution requirements propagating to the model.

---

## Verification

### Verify File Existence

```bash
# Check corpus exists
ls -lh kssm/data/processed/kssm_corpus.jsonl

# Count Gutenberg source files
ls kssm/data/raw/gutenberg/*.txt | wc -l
# Should output: 96

# Check OpenStax directory (should be empty)
ls -la kssm/data/raw/openstax/
# Should show: total 0 (empty directory)
```

### Verify Token Counts

```bash
# Check metadata
cat data/cache_v3/tokens_train_meta.json
# Should show: {"n_tokens": 21169038}

cat data/cache_v3/tokens_val_meta.json
# Should show: {"n_tokens": 1064119}

# Verify corpus line count
wc -l kssm/data/processed/kssm_corpus.jsonl
# Should show: 21312
```

### Verify Public Domain Status

All 96 texts are from Project Gutenberg's public domain catalog:
- Published before 1928 (automatic US public domain)
- No copyright notices in downloaded files
- Gutenberg license allows unrestricted use

**No copyrighted content** (OpenStax CC BY 4.0 content was not downloaded).

---

## What Was NOT Included (Transparency)

### Planned but Not Implemented

**OpenStax Textbooks** (10 textbooks defined in `build_corpus.py`):
- Psychology 2e
- American Government 3e
- Introduction to Sociology 3e
- US History
- World History Volume 1
- Introduction to Philosophy
- Writing Guide
- Principles of Economics 3e
- Biology 2e
- Astronomy 2e

**Status**: Code exists to download these, but they were never executed. The `data/raw/openstax/` directory is empty.

**License if included**: Would have been CC BY 4.0 (requires attribution)

**Impact**: Corpus is simpler (100% Public Domain), no attribution requirements, but lacks modern textbook content.

### Missing Gutenberg Books

**Planned**: 101 books defined in `GUTENBERG_BOOKS` list
**Actual**: 96 books successfully downloaded
**Missing**: 5 books (likely download failures or filtered during cleaning)

---

## Ethical Considerations

### Data Provenance

**All sources verified**:
- ✅ Project Gutenberg texts confirmed public domain
- ✅ No copyrighted content included
- ✅ No CC-licensed content requiring attribution
- ✅ No web-scraped content of uncertain provenance
- ✅ No user-generated content

### Bias Awareness

**Historical bias**: Project Gutenberg corpus skews toward:
- Western (primarily English) literature
- Pre-1928 works (copyright cutoff)
- "Classic" literature (elite/educated authors)
- Religious/philosophical texts (theological bias)

**Not representative of**:
- Modern language use
- Diverse cultural perspectives
- Contemporary issues
- Informal/conversational language
- Internet/social media discourse

**Impact on model**: May produce archaic language patterns, theological framing, Western cultural assumptions.

### Privacy

**No personal data**: All texts are published literary/philosophical works, no user data, no personally identifiable information.

---

## Reproducibility

### Rebuilding the Corpus

```bash
# From repository root
python3 kssm/build_corpus.py

# This will:
# 1. Download 96 Project Gutenberg texts
# 2. Clean and chunk text
# 3. Create kssm/data/processed/kssm_corpus.jsonl
# 4. Takes ~10-15 minutes depending on network speed
```

**Note**: Download order and availability may vary. Project Gutenberg occasionally has server issues or removes/adds texts.

### Tokenizing the Corpus

```bash
# Tokenize into memory-mapped arrays
# (Performed automatically during training startup)
python3 kssm/train_kssm_v3.py --max-steps 10000
```

This creates:
- `data/cache_v3/tokens_train.npy`
- `data/cache_v3/tokens_train_meta.json`
- `data/cache_v3/tokens_val.npy`
- `data/cache_v3/tokens_val_meta.json`

---

## Changes from Original Plan

### Original Design (`build_corpus.py` comments)

```python
"""
Sources (all clean licenses):
1. Standard Ebooks (CC0 - public domain) - ~150M tokens of curated novels
2. OpenStax (CC BY 4.0) - ~50M tokens of textbooks

Target: ~200M tokens total
"""
```

### Actual Implementation

**Sources**: Only Project Gutenberg (Public Domain)
**Tokens**: 22.2M tokens (11% of original 200M target)
**License**: 100% Public Domain (simpler than planned mix)

### Why the Deviation?

**Positive**:
- Simpler licensing (no CC BY 4.0 attribution requirements)
- Faster download/processing (fewer sources)
- Adequate for proof-of-concept training (22M tokens sufficient)

**Trade-offs**:
- Smaller corpus (22M vs 200M target)
- Less diversity (no modern textbooks)
- Historical bias (pre-1928 cutoff)

---

## Future Corpus Improvements

### Potential Additions

1. **OpenStax textbooks** (10 books, ~10-15M tokens, CC BY 4.0)
   - Adds modern educational content
   - Requires attribution in model card

2. **Standard Ebooks** (~150M tokens, CC0)
   - Curated public domain novels
   - Higher quality formatting than raw Gutenberg

3. **Wikipedia philosophy articles** (CC BY-SA 3.0)
   - Modern philosophical discourse
   - Requires share-alike propagation

4. **arXiv philosophy papers** (varied licenses)
   - Contemporary academic philosophy
   - License checking required per-paper

### Scaling Considerations

**Current**: 22M tokens adequate for 10K step experiment
**For production**: 100M+ tokens recommended for:
- Better generalization
- Reduced overfitting
- Broader vocabulary coverage
- Modern language patterns

---

## Citation

When referencing this corpus:

```bibtex
@dataset{kssm_v3_corpus_2026,
  title={K-SSM v3 Training Corpus},
  author={Vasquez, Anthony J., Sr.},
  year={2026},
  note={96 Project Gutenberg texts, 22.2M tokens, Public Domain},
  url={https://github.com/templetwo/liminal-k-ssm}
}
```

**License**: Public Domain (all source texts)
**Attribution**: Not legally required (Public Domain), but appreciated

---

## Contact

**Questions about corpus construction**: See `kssm/build_corpus.py` implementation
**Issues with reproduction**: GitHub Issues
**Licensing questions**: All sources confirmed Public Domain, no restrictions

---

**The corpus is fully transparent. All sources verified Public Domain. No hidden data.** 🌀

*Last Updated: 2026-01-30*
*Corpus Built: 2026-01-29*
*Verified By: Claude Sonnet 4.5*
