# Corpus Expansion Status - Path to 200M Tokens

**Date**: 2026-01-30
**Current Status**: 41M tokens deployed, 95 new books ready

---

## Reality Check: What 200M Actually Requires

### Current State on Mac Studio

| Metric | Value |
|--------|-------|
| **Books downloaded** | 206 |
| **Unique book IDs** | 182 (28 duplicates) |
| **Total tokens** | 41M |
| **Avg tokens/book** | ~199,000 |

### Token Math

```
Current:      41M tokens from 206 books
Target:      200M tokens
Gap:         159M tokens needed
Books/token: ~199k per book average

Required:    159M / 199k = 799 more books
Total:       206 + 799 = 1,005 unique books for 200M
```

**The original plan estimated 470 books for 200M.** This assumed ~425k tokens/book, but classic literature averages only ~200k tokens per book in practice.

---

## Expansion Options

### Option A: Quick Win → 60M Tokens (READY NOW)

**Status**: 95 new English books curated and ready to deploy

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Books | 206 | 301 | +46% |
| Tokens | 41M | ~60M | +46% |
| vs Original 22M | 1.9x | 2.7x | +43% |

**Deployment**:
```bash
# Ready to run NOW
bash kssm/deploy_expansion_60m.sh
```

**Time**: ~15 minutes (4 min download + 6 min process + 5 min tokenize)

**Books Added** (95 total):
- 40 more British/American literature
- 20 more philosophy works (English translations)
- 15 more science/nature writing
- 10 more ancient classics (English translations)
- 10 more American historical/political works

**Pros**:
- ✅ Ready immediately
- ✅ 2.7x original corpus size
- ✅ Substantial improvement for training
- ✅ All Public Domain, English-language

**Cons**:
- Only 30% of 200M target

---

### Option B: Medium Goal → 100M Tokens

**Status**: Requires curation of 296 more English books

| Metric | Value |
|--------|-------|
| Total books needed | ~502 |
| Currently have | 206 |
| Ready to deploy | 95 |
| Still need | ~201 more books |

**What's needed**:
1. Deploy the 95 ready books → 60M tokens
2. Curate 201 more English public domain books
3. Download and process → 100M total

**Effort**:
- Curation: ~3-4 hours to find quality English PD books
- Download: ~8 minutes (201 books × 2.5s)
- Processing: ~10 minutes
- Total: ~4 hours

---

### Option C: Original Goal → 200M Tokens

**Status**: Requires curation of 799 more English books

| Metric | Value |
|--------|-------|
| Total books needed | ~1,005 |
| Currently have | 206 |
| Ready to deploy | 95 |
| Still need | ~704 more books |

**Effort**: ~15-20 hours curation time

---

## Recommendation: Incremental Approach

**Phase 1: Deploy 60M NOW** ⭐
```bash
bash kssm/deploy_expansion_60m.sh
```

**Phase 2: Train and Assess**
- Does 60M improve over 41M?
- Richer R-space exploration?
- Higher quality samples?

**Phase 3: Decision Point**
- Strong gains → Continue to 100M or 200M
- Diminishing returns → Stop at 60M

---

## Commands Ready

**Deploy 60M now**:
```bash
bash kssm/deploy_expansion_60m.sh
```

**Then start training**:
```bash
ssh tony_studio@192.168.1.195
cd ~/liminal-k-ssm
# Update train script to use cache_v3_60m
python3 kssm/train_kssm_v3.py --max-steps 100
```

---

**Decision needed**: Deploy 60M now? Skip to 100M? Commit to full 200M?
