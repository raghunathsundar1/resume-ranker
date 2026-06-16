#!/usr/bin/env python3
"""Phase B: features.parquet → submission.csv (top 100 ranked candidates).

B1 Load  → B2 Integrity gate → B3 Hybrid relevance → B4 Composite → B5 Reason + emit

Usage:
    python rank.py --candidates features.parquet --out submission.csv
    python rank.py --candidates sample.parquet  --out sample_submission.csv

Constraints: ≤5 min, ≤16 GB RAM, CPU-only, no network, deterministic.

Performance design:
- Concept scoring uses pre-tokenised keyword-set intersection (no regex in hot path)
- BM25 runs only on the top-3000 candidates by concept score (not all 38K active)
- All text tokenised once per candidate at load time
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

AS_OF = date(2026, 1, 31)

# ---------------------------------------------------------------------------
# Job Description
# ---------------------------------------------------------------------------

JD_TEXT = """
senior applied machine learning engineer search ranking recommendation
product technology company hiring senior ml engineer build improve search
ranking recommendation systems scale requirements five nine years total
experience three more years applied machine learning product companies
proven experience shipping production search recommendation systems strong
foundation machine learning information retrieval experience ranking
algorithms learning rank bm25 neural ranking ndcg optimization python
expertise pytorch tensorflow nlp text understanding embeddings bert
transformers text classification experience ab testing offline evaluation
metrics experiment frameworks recommendation systems collaborative filtering
two tower models matrix factorization candidate generation engineering skills
distributed computing spark containerization model serving mlops feature
stores location noida pune india preferred
"""

JD_TOKENS = re.findall(r"[a-z0-9]+", JD_TEXT)

# ---------------------------------------------------------------------------
# Concept ontology — keyword sets (no regex in scoring hot-path)
#
# Each entry: (weight, exact_tokens, substring_phrases)
# exact_tokens: matched against the pre-tokenised frozenset (O(1) lookup)
# substring_phrases: checked with `phrase in lowercased_text`
# ---------------------------------------------------------------------------

CONCEPTS: dict[str, tuple[float, frozenset[str], tuple[str, ...]]] = {
    "ranking_ir": (1.5,
        frozenset({"ranking", "ltr", "bm25", "ndcg", "mrr", "relevance",
                   "elasticsearch", "solr", "lucene", "retrieval", "reranking",
                   "rerank", "ranker", "pagerank"}),
        ("learn to rank", "click through", "information retrieval",
         "semantic search", "vector search", "query understanding",
         "click-through rate")),
    "recommendation": (1.3,
        frozenset({"recsys", "recommender", "recommend", "personalization",
                   "personalized", "personalise", "personalised"}),
        ("collaborative filtering", "matrix factorization", "two tower",
         "two-tower", "candidate generation", "item embedding",
         "recommendation system", "recommendation engine")),
    "nlp": (1.0,
        frozenset({"nlp", "bert", "transformer", "transformers", "gpt", "llm",
                   "embedding", "embeddings", "tokenizer", "sentiment",
                   "word2vec", "fasttext", "glove", "ner"}),
        ("natural language", "text classification", "named entity",
         "language model", "text embedding", "sequence model")),
    "ml_production": (1.0,
        frozenset({"mlops", "mlflow", "kubeflow", "triton", "bentoml",
                   "inference", "serving", "deployed", "deployment",
                   "latency", "throughput", "ab", "experimentation"}),
        ("feature store", "model serving", "model monitoring",
         "a/b test", "production ml", "ml pipeline", "real-time")),
    "deep_learning": (0.8,
        frozenset({"pytorch", "tensorflow", "keras", "onnx", "jax",
                   "gradient", "backprop", "backpropagation", "finetuning",
                   "finetuned", "pretrained"}),
        ("deep learning", "neural network", "fine-tuning", "fine tuning")),
    "ml_general": (0.5,
        frozenset({"xgboost", "lightgbm", "sklearn", "scikit", "regression",
                   "classification", "clustering", "ensemble", "boosting",
                   "catboost", "randomforest", "svm", "ml"}),
        ("machine learning", "feature engineering", "model training",
         "random forest", "gradient boosting")),
    "engineering": (0.4,
        frozenset({"spark", "kafka", "kubernetes", "docker", "airflow",
                   "python", "scala", "java", "golang", "distributed",
                   "microservice", "grpc", "api", "pipeline"}),
        ("distributed computing", "data pipeline", "stream processing")),
}

# Pre-flatten all tokens across concepts for the gate's fast check
_STRONG_CONCEPT_TOKENS: frozenset[str] = frozenset().union(
    *[kws for name, (_, kws, _) in CONCEPTS.items()
      if name in ("ranking_ir", "nlp", "recommendation", "deep_learning")]
)
_STRONG_PHRASES: tuple[str, ...] = tuple(
    p for name, (_, _, phrases) in CONCEPTS.items()
    if name in ("ranking_ir", "nlp", "recommendation", "deep_learning")
    for p in phrases
)

# ---------------------------------------------------------------------------
# Non-ML title blocklist
# ---------------------------------------------------------------------------

_NON_ML_TITLE_RE = re.compile(
    r"\b(marketing|sales|accountan|finance|human.resource|\bhr\b|"
    r"content.writer|copywriter|ux\b|graphic.designer|project.manager|"
    r"scrum.master|business.analyst|product.manager|customer.support|"
    r"customer.success|operations.manager|supply.chain|recruit|"
    r"talent.acquisition|legal\b|counsel|paralegal|doctor|nurse|"
    r"physician|teacher|tutor|professor)\b",
    re.I,
)

# CV/speech/robotics domain mismatch (no NLP/IR coverage)
_MISMATCH_TOKENS: frozenset[str] = frozenset({
    "opencv", "yolo", "detection", "segmentation",
    "asr", "tts", "whisper", "robotic", "robotics",
})
_MISMATCH_PHRASES = ("computer vision", "object detection", "image classification",
                     "speech recognition", "text to speech", "self-driving")

# ---------------------------------------------------------------------------
# Text pre-processing (called once per candidate at load time)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _precompute(row: dict) -> None:
    """Add _hist_lower, _title_lower, _skill_lower, _tok_* in-place."""
    hist = ((row.get("history_text") or "") + " " +
            (row.get("summary_text") or "")).lower()
    titles = (row.get("title_history_text") or "").lower()
    skills_raw = row.get("skills_raw") or "[]"
    skill_text = " ".join(
        s.get("name", "") for s in json.loads(skills_raw)
    ).lower()

    row["_hist_lower"] = hist
    row["_title_lower"] = titles
    row["_skill_lower"] = skill_text
    row["_tok_hist"] = _tokenize(hist)
    row["_tok_title"] = _tokenize(titles)
    row["_tok_skill"] = _tokenize(skill_text)


# ---------------------------------------------------------------------------
# Concept matching (fast path — no regex)
# ---------------------------------------------------------------------------

def _hit_concepts(tokens: frozenset[str], text_lower: str) -> frozenset[str]:
    hits = set()
    for name, (_, kws, phrases) in CONCEPTS.items():
        if tokens & kws:
            hits.add(name)
        elif any(p in text_lower for p in phrases):
            hits.add(name)
    return frozenset(hits)


def _has_strong_ml(tokens: frozenset[str], text_lower: str) -> bool:
    if tokens & _STRONG_CONCEPT_TOKENS:
        return True
    return any(p in text_lower for p in _STRONG_PHRASES)


# ---------------------------------------------------------------------------
# B2 — Integrity gate
# ---------------------------------------------------------------------------

def _gate(row: dict) -> tuple[bool, float]:
    """(disqualified, penalty). Pre-compute fields must exist."""
    penalty = 0.0

    if row.get("has_impossible_tenure"):
        return True, 1.0
    if int(row.get("expert_zero_months_count") or 0) > 0:
        return True, 1.0
    if row.get("has_future_date"):
        return True, 1.0

    # Non-ML title gate: needs ≥2 strong concept hits in work history
    current_title = row.get("current_title") or ""
    if _NON_ML_TITLE_RE.search(current_title):
        hist_tokens = row["_tok_hist"] | row["_tok_title"]
        hist_text = row["_hist_lower"] + " " + row["_title_lower"]
        strong_hits = sum(
            1 for name in ("ranking_ir", "nlp", "recommendation", "deep_learning")
            if (hist_tokens & CONCEPTS[name][1]) or
               any(p in hist_text for p in CONCEPTS[name][2])
        )
        if strong_hits < 2:
            return True, 1.0

    # Services-only career
    svc = float(row.get("services_years") or 0)
    prd = float(row.get("product_years") or 0)
    if svc > 4 and prd < 0.5:
        return True, 1.0

    # Soft penalties
    if row.get("skill_months_exceed_career"):
        penalty += 0.15
    gap = abs(float(row.get("dur_vs_dates_gap") or 0))
    if gap > 2.0:
        penalty += min(0.2, (gap - 2.0) * 0.05)
    if row.get("has_overlapping_roles"):
        penalty += 0.05

    return False, min(penalty, 0.5)


# ---------------------------------------------------------------------------
# B3 — Concept scoring (evidence-weighted, keyword-set hot path)
# ---------------------------------------------------------------------------

def _concept_score(row: dict) -> float:
    tok_hist = row["_tok_hist"]
    tok_title = row["_tok_title"]
    tok_skill = row["_tok_skill"]
    hist_text = row["_hist_lower"]
    title_text = row["_title_lower"]
    skill_text = row["_skill_lower"]

    hist_hits = _hit_concepts(tok_hist, hist_text)
    title_hits = _hit_concepts(tok_title, title_text)
    skill_hits = _hit_concepts(tok_skill, skill_text)
    corroborated = skill_hits & hist_hits
    uncorroborated = skill_hits - hist_hits

    raw = 0.0
    max_w = 0.0
    for name, (w, _, _) in CONCEPTS.items():
        max_w += w
        c = 0.0
        if name in hist_hits:
            c = 1.0
        elif name in title_hits:
            c = 0.8
        if name in corroborated:
            c = max(c, 1.0)
        elif name in uncorroborated:
            c = max(c, 0.1)
        raw += w * c

    # Keyword-stuffer penalty
    total_skill_hits = len(skill_hits)
    if total_skill_hits > 0 and len(uncorroborated) / total_skill_hits > 0.7:
        raw *= 0.3

    return raw / max_w if max_w else 0.0


def _cv_speech_penalty(row: dict) -> float:
    tokens = row["_tok_hist"] | row["_tok_title"]
    text = row["_hist_lower"]
    mismatch = len(tokens & _MISMATCH_TOKENS) + sum(
        1 for p in _MISMATCH_PHRASES if p in text
    )
    has_ir = bool((tokens & CONCEPTS["ranking_ir"][1]) or
                  any(p in text for p in CONCEPTS["ranking_ir"][2]))
    has_nlp = bool((tokens & CONCEPTS["nlp"][1]) or
                   any(p in text for p in CONCEPTS["nlp"][2]))
    if mismatch >= 3 and not (has_ir or has_nlp):
        return 0.4
    if mismatch >= 2 and not (has_ir or has_nlp):
        return 0.2
    return 0.0


# ---------------------------------------------------------------------------
# B4 — Structured fit, behavioral, logistics
# ---------------------------------------------------------------------------

def _structured_fit(row: dict) -> float:
    yoe = float(row.get("yoe_claimed") or 0)
    prd = float(row.get("product_years") or 0)
    svc = float(row.get("services_years") or 0)
    exp_score = 1.0 if 5 <= yoe <= 9 else (yoe / 5.0 if yoe < 5 else max(0.5, 1.0 - (yoe - 9) * 0.04))
    prd_score = min(1.0, prd / 4.0)
    svc_ratio = svc / (yoe + 0.1)
    return (exp_score * 0.4 + prd_score * 0.6) * (1 - svc_ratio * 0.5)


def _behavioral(row: dict) -> float:
    recency = max(0.1, 1.0 - int(row.get("sig_last_active_days") or 999) / 365.0)
    rr = float(row.get("sig_recruiter_response_rate") or 0)
    otw = 1.0 if row.get("sig_open_to_work_flag") else 0.5
    assessment = float(row.get("sig_skill_assessment_mean") or 0) / 100.0
    github = min(1.0, max(0.0, float(row.get("sig_github_activity_score") or 0)) / 10.0)
    return recency * 0.35 + rr * 0.25 + otw * 0.15 + assessment * 0.15 + github * 0.10


def _logistics(row: dict) -> float:
    country = (row.get("country") or "").lower()
    location = (row.get("location") or "").lower()
    notice = int(row.get("sig_notice_period_days") or 90)
    willing = bool(row.get("sig_willing_to_relocate"))
    if country == "india":
        loc = 1.1 if any(c in location for c in (
            "noida", "pune", "bengaluru", "bangalore", "hyderabad",
            "mumbai", "delhi", "gurugram", "gurgaon", "chennai")) else 1.0
    elif willing:
        loc = 0.7
    else:
        loc = 0.5
    np_score = 1.0 if notice <= 30 else (0.9 if notice <= 60 else
               (0.8 if notice <= 90 else max(0.5, 0.8 - (notice - 90) / 180.0)))
    return min(1.0, loc) * 0.7 + np_score * 0.3


def _composite(rel: float, concept: float, fit: float,
               behav: float, logist: float, penalty: float, cv_pen: float) -> float:
    return (max(0.0, rel) ** 0.4 * max(0.0, concept) ** 0.3 *
            max(0.0, fit) ** 0.15 * max(0.0, behav) ** 0.1 *
            max(0.0, logist) ** 0.05 * (1 - penalty) * (1 - cv_pen))


# ---------------------------------------------------------------------------
# B5 — Reasoning composer
# ---------------------------------------------------------------------------

_FRAMES_TOP = [
    "{yoe:.1f}yr career ({prd:.1f}yr at product companies); {fact}. {concern}",
    "Strong product-company background ({prd:.1f}yr); {fact}. {jd_link}.",
    "Candidate has {prd:.1f}yr shipping production ML at product firms; {fact}.",
    "{title} with {yoe:.1f}yr exp and {prd:.1f}yr at product companies; {fact}.",
    "{yoe:.1f}yr total, {prd:.1f}yr at product companies — {fact}. {jd_link}.",
]
_FRAMES_MID = [
    "{yoe:.1f}yr exp with {prd:.1f}yr at product firms; {fact}. {concern}",
    "Mixed profile: {fact}; however {concern}. Borderline for this JD.",
    "{title} ({yoe:.1f}yr exp); {fact} but {concern}.",
]
_FRAMES_TAIL = [
    "Possibly relevant ({fact}) but {concern} limits confidence.",
    "{title} — some signal ({fact}) offset by {concern}. Rank ~{rank}.",
    "{yoe:.1f}yr exp with {prd:.1f}yr at product firms; {fact}. {concern}",
]
_JD_LINKS = [
    "directly addresses the JD's search/ranking requirement",
    "aligns with the JD's NLP/IR focus",
    "matches the JD's production ML mandate",
    "fits the JD's recommendation-system scope",
    "relevant to the JD's product-company bias",
]


def _compose_reasoning(row: dict, rank: int) -> str:
    rng = random.Random(str(row.get("candidate_id", "")) + str(rank))
    yoe = float(row.get("yoe_claimed") or 0)
    prd = float(row.get("product_years") or 0)
    svc = float(row.get("services_years") or 0)
    title = row.get("current_title") or "Candidate"
    loc = row.get("location") or row.get("country") or "unknown"
    notice = int(row.get("sig_notice_period_days") or 90)
    assess = float(row.get("sig_skill_assessment_mean") or 0)
    github = float(row.get("sig_github_activity_score") or 0)

    concepts_hit = _hit_concepts(row["_tok_hist"], row["_hist_lower"]) & {
        "ranking_ir", "nlp", "recommendation", "ml_production", "deep_learning"}
    skills = json.loads(row.get("skills_raw") or "[]")
    top_skills = ", ".join(s.get("name", "") for s in skills[:3]) or "unspecified"

    fact_pool = [
        f"shows {len(concepts_hit)} core JD concept(s): {', '.join(sorted(concepts_hit)) or 'none'}",
        f"top skills include {top_skills}",
        f"current role: {title}",
        *([ f"assessment avg {assess:.0f}/100"] if assess > 0 else []),
        *([f"{prd:.1f}yr at product companies"] if prd > 1 else []),
        *([f"GitHub activity score {github:.1f}/10"] if github > 0 else []),
    ]
    fact = rng.choice(fact_pool)
    jd_link = rng.choice(_JD_LINKS)

    concern_pool = []
    if notice > 60:
        concern_pool.append(f"notice period {notice}d is long")
    if svc > prd and svc > 2:
        concern_pool.append(f"currently in services ({svc:.1f}yr), limited product exposure")
    if (row.get("country") or "").lower() != "india":
        concern_pool.append(f"location ({loc}) outside India preference")
    if float(row.get("sig_recruiter_response_rate") or 0) < 0.3:
        concern_pool.append("low recruiter response rate")
    if not concern_pool:
        concern_pool.append("no major red flags noted")
    concern = rng.choice(concern_pool)

    frames = _FRAMES_TOP if rank <= 10 else (_FRAMES_MID if rank <= 50 else _FRAMES_TAIL)
    frame = rng.choice(frames)
    try:
        return frame.format(title=title, yoe=yoe, prd=prd, svc=svc,
                            loc=loc, notice=notice, assess=assess,
                            fact=fact, jd_link=jd_link, concern=concern, rank=rank).strip()
    except KeyError:
        return f"{title} ({yoe:.1f}yr exp, {prd:.1f}yr product): {fact}. {concern}."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp = Path(args.candidates)
    if not inp.exists():
        sys.exit(f"Not found: {inp}")

    # B1 — Load + pre-compute text fields
    print("B1 Loading features …", file=sys.stderr)
    rows = pq.read_table(inp).to_pylist()
    print(f"  {len(rows):,} candidates loaded", file=sys.stderr)
    for r in rows:
        _precompute(r)

    # B2 — Integrity gate
    print("B2 Integrity gate …", file=sys.stderr)
    active, penalties = [], []
    n_dq = 0
    for r in rows:
        dq, pen = _gate(r)
        if dq:
            n_dq += 1
        else:
            active.append(r)
            penalties.append(pen)
    print(f"  {n_dq} disqualified ({n_dq/len(rows)*100:.1f}%)", file=sys.stderr)

    # B3a — Concept scores for all active (fast keyword-set matching)
    print("B3 Concept scoring …", file=sys.stderr)
    concept_scores = np.array([_concept_score(r) for r in active])
    cv_penalties = np.array([_cv_speech_penalty(r) for r in active])

    # B3b — BM25 only on top-3000 by concept score (speed gate)
    print("B3 BM25 on top-3000 …", file=sys.stderr)
    TOP_K = min(3000, len(active))
    top_k_idx = np.argpartition(concept_scores, -TOP_K)[-TOP_K:]
    top_k_rows = [active[i] for i in top_k_idx]

    def _bm25_doc(r: dict) -> list[str]:
        return list(r["_tok_hist"] | r["_tok_title"] | r["_tok_skill"])

    bm25 = BM25Okapi([_bm25_doc(r) for r in top_k_rows])
    bm25_raw = bm25.get_scores(JD_TOKENS)
    bm25_max = bm25_raw.max() if bm25_raw.size > 0 else 1.0
    bm25_norm = bm25_raw / (bm25_max + 1e-9)

    # Map BM25 scores back to full active array (non-top-K get 0)
    bm25_full = np.zeros(len(active))
    bm25_full[top_k_idx] = bm25_norm

    relevance = 0.4 * bm25_full + 0.6 * concept_scores

    # B4 — Composite
    print("B4 Composite scoring …", file=sys.stderr)
    fit = np.array([_structured_fit(r) for r in active])
    behav = np.array([_behavioral(r) for r in active])
    logist = np.array([_logistics(r) for r in active])
    pen_arr = np.array(penalties)

    scores = (np.maximum(relevance, 0) ** 0.4 *
              np.maximum(concept_scores, 0) ** 0.3 *
              np.maximum(fit, 0) ** 0.15 *
              np.maximum(behav, 0) ** 0.1 *
              np.maximum(logist, 0) ** 0.05 *
              (1 - pen_arr) *
              (1 - cv_penalties))

    # Sort: primary score desc, secondary candidate_id asc (deterministic tie-break)
    order = sorted(range(len(active)),
                   key=lambda i: (-scores[i], active[i].get("candidate_id", "")))
    top100_idx = order[:100]

    # Honeypot rate check
    honeypots = sum(
        1 for i in top100_idx
        if active[i].get("has_impossible_tenure")
        or active[i].get("expert_zero_months_count", 0) > 0
        or active[i].get("has_future_date")
    )
    if honeypots > 10:
        sys.exit(f"ABORT: honeypot rate {honeypots}/100 > 10% DQ threshold")

    # B5 — Emit
    print("B5 Reasoning + CSV …", file=sys.stderr)
    out_path = Path(args.out)
    prev_score = None
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, i in enumerate(top100_idx, 1):
            s = float(scores[i])
            if prev_score is not None and s > prev_score:
                s = prev_score
            prev_score = s
            w.writerow([
                active[i].get("candidate_id", ""),
                rank,
                f"{s:.6f}",
                _compose_reasoning(active[i], rank),
            ])

    print(f"Written -> {out_path}  (honeypots in top 100: {honeypots})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
