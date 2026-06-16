# Redrob Ranker

Rank 100,000 candidates against a job description and output the top 100 as CSV,
each with a short reasoning. Hackathon project; judged offline against a hidden
ground truth. The dataset is adversarial (keyword stuffers, plain-language strong
candidates, behavioral twins, ~80 impossible "honeypot" profiles).

## The non-negotiable constraints

The **ranking step** (`rank.py`) must run in **≤5 min, ≤16 GB RAM, CPU-only, with
no network**. Anything expensive (embeddings, ontology build) goes in offline
precompute, never in the rank path. Output must be **deterministic** — same input,
byte-identical CSV (judges reproduce the run). No `datetime.now()`; use the pinned
`AS_OF` constant. Seed any randomness per-candidate.

**Honeypots > 10% of the top 100 = automatic disqualification.** Reading profiles
for consistency is a pass/fail requirement, not polish.

## Layout

- `normalize.py` — Phase A1: raw JSONL → `features.parquet`. The only place raw
  field names live (the `*_MAP` tables at the top). **DONE.**
- `rank.py` — the single command that produces the submission. **Not built yet.**
- `docs/` — detailed references; read the relevant one before working on that area.

## Build & run

```bash
python normalize.py --input candidates.jsonl.gz --out features.parquet
python normalize.py --input sample_candidates.json --out sample.parquet --json-array  # sample is a JSON array
python validate_submission.py submission.csv   # always run before "done"
```

Deps: `orjson`, `python-dateutil`, `pyarrow`, plus (later) `onnxruntime`, `rank_bm25`.

## How to work here

- **Verify changes by running them.** After editing parsing or scoring, run on
  `sample_candidates.json` and eyeball the affected columns. There is no leaderboard,
  so correctness is established locally — see `docs/evaluation.md`.
- **Commit per logical step with a message that says what changed and why** (cite
  local eval deltas when relevant). Judges grade git history for real iteration vs.
  a single dump; never squash the meaningful history. Details: `docs/judging.md`.
- **Prefer the simplest defensible mechanism.** Finalists defend every choice in a
  live interview, so favor transparent, explainable logic over opaque models.

## Reference docs — read the one that matches your task before starting

- `docs/architecture.md` — the two-phase pipeline and how the stages connect.
- `docs/schema.md` — the real candidate record structure and the 23 signals.
- `docs/scoring.md` — relevance, composite gating, evidence-weighted skills.
- `docs/integrity.md` — honeypot + JD-disqualifier checks (what's hard vs. soft).
- `docs/reasoning.md` — the Stage-4 reasoning rubric the composer must satisfy.
- `docs/judging.md` — the 5 evaluation stages, submission format, git-history rules.
- `docs/evaluation.md` — building the local hand-labeled eval set and metrics.
- `docs/gotchas.md` — non-obvious findings from testing; read before touching parsing.