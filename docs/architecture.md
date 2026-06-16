# Architecture

Two phases. Everything expensive lives in A (offline, no limits). The rank path (B)
is a sequence of vectorized numpy ops over precomputed artifacts — nothing in B
scales with per-candidate model inference.

## Phase A — offline precompute (GPU/network OK; outputs committed as artifacts)

- **A1 normalize** — raw JSONL → `features.parquet`. Built: `normalize.py`.
- **A2 ontology + skill scorer** — map raw skill/text vocab onto canonical concepts;
  emit claimed-vs-evidenced concept matrices. Pending. Inputs already produced by A1:
  `skills_raw`, `history_text`, `summary_text`.
- **A3 company classification** — employer → services / product / research. Partial
  (services detection lives in `normalize.py`); needs the full lookup table + the
  product/research split.
- **A4 embeddings** — facet text → float16 `.npy` matrices via a small sentence
  model exported to ONNX/int8. Pending. ~75 MB per facet at 100K×384; mmap at load.

## Phase B — the ranking step (`python rank.py --candidates ... --out submission.csv`)

1. **B1 load** — artifacts + pool, streamed (`orjson` + gzip). Embeddings mmap'd.
2. **B2 integrity gate** — vectorized honeypot + JD-disqualifier masks. See `integrity.md`.
3. **B3 hybrid relevance** — dense cosine (one runtime JD embed via ONNX-CPU) blended
   with evidence-weighted concept/BM25 score. See `scoring.md`.
4. **B4 composite** — multiplicative gating of relevance × fit × behavioral × logistics.
5. **B5 rerank + reason + emit** — top ~300 → deep per-candidate checks → top 100 →
   deterministic reasoning → self-validated CSV. See `reasoning.md`.

## Why no vector DB

One JD query against 100K vectors is a single numpy matmul (~ms). FAISS/Qdrant solve
repeated-query indexing we don't have, add a dependency, and are a liability to defend
in the interview. Don't add one.

## Why no learned ranking model

No labels + a hidden test query → a learning-to-rank model would overfit a small
hand-labeled set, hurt reproducibility, and be hard to explain. We use a transparent,
hand-tuned feature formula. This is a deliberate, defensible choice.