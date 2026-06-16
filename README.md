---
title: Redrob Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Redrob Ranker

Ranks 100,000 ML-job candidates against a job description and outputs the top 100
as a CSV with per-candidate reasoning. Hackathon project.

## Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Normalize raw candidates → feature parquet (Phase A1)
python normalize.py --input candidates.jsonl.gz --out features.parquet
# For the JSON-array sample:
# python normalize.py --input data/sample_candidates.json --out sample.parquet --json-array

# 3. Rank and emit top 100 (Phase B) — ≤5 min, CPU-only, no network
python rank.py --candidates features.parquet --out submission.csv

# 4. Validate before submitting
python validate_submission.py submission.csv
```

## Architecture

```
candidates.jsonl.gz
       │
  normalize.py (A1) ──► features.parquet
                                │
                           rank.py (B1-B5)
                                │
                          submission.csv
```

**Phase A1 (normalize.py):** Parses raw JSONL, classifies industries
(services / product / research), computes 62 features including integrity
signals and text blobs for BM25/concept scoring.

**Phase B (rank.py):**
- B2 Integrity gate: hard-excludes honeypots (impossible tenure, expert-0-months,
  future dates) and JD disqualifiers (services-only career, non-ML roles)
- B3 Hybrid relevance: 40% BM25 + 60% evidence-weighted concept scoring.
  Concept scoring multiplies by source: work-history evidence (1.0×), titles (0.8×),
  corroborated skills (1.0×), uncorroborated skill-list claims (0.1×). This collapses
  keyword stuffers while rewarding plain-language strong candidates.
- B4 Composite: `relevance^0.4 · concept^0.3 · fit^0.15 · behavioral^0.1 · logistics^0.05 · (1−penalty)`
- B5 Reasoning: deterministic per-candidate reasoning from parsed fields only
  (seeded by candidate_id for reproducibility; no LLM).

## Constraints met

| Constraint | Status |
|---|---|
| ≤5 min rank step | ✓ CPU BM25 + numpy ops |
| ≤16 GB RAM | ✓ parquet loaded once, no embeddings |
| CPU-only | ✓ no GPU, no ONNX runtime |
| No network | ✓ JD embedded as constant |
| Deterministic | ✓ AS_OF pinned, reasoning seeded by candidate_id |
| Honeypots < 10% | ✓ gate self-checks and aborts if exceeded |

## Dependencies

```
orjson, python-dateutil, pyarrow, rank-bm25, numpy, tqdm
```
