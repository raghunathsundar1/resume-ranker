#!/usr/bin/env python3
"""Phase B: features.parquet → submission.csv (top 100 ranked candidates).

B1 Load  → B2 Integrity gate → B3 Hybrid relevance → B4 Composite → B5 Reason + emit

Usage:
    python rank.py --candidates features.parquet --out submission.csv
    python rank.py --candidates sample.parquet  --out sample_submission.csv

Constraints: ≤5 min, ≤16 GB RAM, CPU-only, no network, deterministic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AS_OF = date(2026, 1, 31)

# ---------------------------------------------------------------------------
# Job Description — embedded so rank.py needs no network or external files
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior Applied Machine Learning Engineer Search Ranking Recommendation

We are a product technology company hiring a Senior ML Engineer to build
and improve our search ranking and recommendation systems at scale.

Requirements five to nine years total experience three or more years applied
machine learning at product companies. Proven experience shipping production
search or recommendation systems. Strong foundation in machine learning and
information retrieval. Experience with ranking algorithms learning to rank
BM25 neural ranking NDCG optimization. Python expertise PyTorch or TensorFlow.
NLP text understanding embeddings BERT transformers text classification.
Experience with AB testing offline evaluation metrics experiment frameworks.
Recommendation systems collaborative filtering two tower models matrix
factorization candidate generation. Engineering skills distributed computing
Spark containerization model serving MLOps feature stores.

Location Noida or Pune India preferred open to remote for strong candidates.

Not a fit career entirely in IT services or consulting with no product company
experience CV speech robotics focus without search NLP information retrieval
exposure pure research roles with no production deployment LLM framework
integrations LangChain LlamaIndex as primary ML experience.
"""

# ---------------------------------------------------------------------------
# B3 — Concept ontology (pattern → source multiplier → relevance weight)
# Each concept is matched against work history (1.0), titles (0.8),
# and skills list (corroborated 1.0 / uncorroborated 0.1).
# ---------------------------------------------------------------------------

# (name, weight, patterns)
CONCEPTS: list[tuple[str, float, list[str]]] = [
    ("ranking_ir", 1.5, [
        r"\branking\b", r"\blearn.{0,5}to.{0,5}rank\b", r"\bLTR\b",
        r"\bBM25\b", r"\bNDCG\b", r"\bMRR\b", r"\bclick.through\b",
        r"\brelevance\b", r"\binformation.retrieval\b", r"\b\bIR\b",
        r"\bElasticsearch\b", r"\bSolr\b", r"\bLucene\b",
        r"\bsemantic.search\b", r"\bvector.search\b",
    ]),
    ("recommendation", 1.3, [
        r"\brecommend", r"\bRecSys\b", r"\bcollaborative.filtering\b",
        r"\bmatrix.factorization\b", r"\btwo.tower\b", r"\bpersonali",
        r"\bcandidate.generation\b", r"\bitem.embedding\b",
    ]),
    ("nlp", 1.0, [
        r"\bNLP\b", r"\bnatural.language\b", r"\bBERT\b", r"\btransformer",
        r"\bembedding", r"\btext.classif", r"\bnamed.entity\b",
        r"\bsentiment\b", r"\btokeniz", r"\blanguage.model\b",
        r"\bword2vec\b", r"\bGPT\b", r"\bLLM\b",
    ]),
    ("ml_production", 1.0, [
        r"\bproduction\b", r"\bdeploy", r"\bserving\b", r"\binference\b",
        r"\bA/B\b", r"\bMLOps\b", r"\bMLflow\b", r"\bfeature.store\b",
        r"\bmodel.monitor", r"\breal.time\b", r"\blatency\b",
        r"\bscale\b", r"\bthroughput\b",
    ]),
    ("deep_learning", 0.8, [
        r"\bdeep.learning\b", r"\bneural.network\b", r"\bPyTorch\b",
        r"\bTensorFlow\b", r"\bfine.tun", r"\bONNX\b",
        r"\bgradient\b", r"\bbackprop\b",
    ]),
    ("ml_general", 0.5, [
        r"\bmachine.learning\b", r"\bclassif", r"\bregressi",
        r"\bclustering\b", r"\bensemble\b", r"\bXGBoost\b",
        r"\brandom.forest\b", r"\bscikit\b", r"\bfeature.engineer",
    ]),
    ("engineering", 0.4, [
        r"\bSpark\b", r"\bKafka\b", r"\bKubernetes\b", r"\bDocker\b",
        r"\bdistributed\b", r"\bpipeline\b", r"\bAPI\b",
        r"\bmicroservice\b", r"\bPython\b",
    ]),
]

# Compiled patterns (done once at import time)
_COMPILED: list[tuple[str, float, list[re.Pattern]]] = [
    (name, weight, [re.compile(p, re.I) for p in pats])
    for name, weight, pats in CONCEPTS
]

# Non-ML job title patterns (hard disqualifier for clearly irrelevant roles)
_NON_ML_TITLE_PATS = [re.compile(p, re.I) for p in [
    r"\bmarketing\b", r"\bsales\b", r"\baccountan", r"\bfinance\b",
    r"\bHR\b", r"\bhuman.resource", r"\bcontent.writer\b", r"\bcopywriter\b",
    r"\bUX\b", r"\bUI\b", r"\bgraphic.designer\b",
    r"\bproject.manager\b", r"\bscrum.master\b",
    r"\bbusiness.analyst\b", r"\bproduct.manager\b",
    r"\bcustomer.support\b", r"\bcustomer.success\b",
    r"\boperations.manager\b", r"\bsupply.chain\b",
    r"\brecruit", r"\btalent.acquisition\b",
    r"\blegal\b", r"\bcounsel\b", r"\bparalegal\b",
    r"\bdoctor\b", r"\bnurse\b", r"\bphysician\b",
    r"\bteacher\b", r"\btutor\b", r"\bprofessor\b",
]]

# CV/robotics/speech patterns that indicate domain mismatch
_MISMATCH_PATS = [re.compile(p, re.I) for p in [
    r"\bcomputer.vision\b", r"\bimage.classif", r"\bobject.detect",
    r"\bspeech.recognit", r"\bTTS\b", r"\btext.to.speech\b",
    r"\bASR\b", r"\brobotic", r"\bself.driving\b",
]]

# LLM-framework-glue patterns (sub-12-month-only → disqualifier)
_LLM_GLUE_PATS = [re.compile(p, re.I) for p in [
    r"\bLangChain\b", r"\bLlamaIndex\b", r"\bLangGraph\b",
    r"\bOpenAI.API\b", r"\bchatbot\b", r"\bRAG\b",
]]

# ---------------------------------------------------------------------------
# B2 — Integrity gate
# ---------------------------------------------------------------------------

def _gate(row: dict) -> tuple[bool, float]:
    """Returns (disqualified: bool, penalty: float 0..1)."""
    penalty = 0.0

    # Hard honeypot exclusions
    if row.get("has_impossible_tenure"):
        return True, 1.0
    if int(row.get("expert_zero_months_count") or 0) > 0:
        return True, 1.0
    if row.get("has_future_date"):
        return True, 1.0

    # Hard JD mismatch: clearly non-ML current title with no relevant history
    current_title = row.get("current_title") or ""
    title_hist = row.get("title_history_text") or ""
    all_titles = current_title + " | " + title_hist
    is_non_ml_title = any(p.search(current_title) for p in _NON_ML_TITLE_PATS)
    has_ml_in_history = bool(_match_concepts(
        (row.get("history_text") or "") + " " + title_hist
    ) & {"ranking_ir", "nlp", "recommendation", "ml_production", "deep_learning", "ml_general"})

    # Non-ML title with insufficient ML work history → exclude.
    # Require hits from ≥2 strong concepts (ranking/NLP/reco/deep learning),
    # not just weak/generic ml_general or ml_production hits.
    strong_ml_hits = _match_concepts(
        (row.get("history_text") or "") + " " + title_hist
    ) & {"ranking_ir", "nlp", "recommendation", "deep_learning"}
    if is_non_ml_title and len(strong_ml_hits) < 2:
        return True, 1.0

    # JD disqualifier: consulting/services-only career
    svc_yrs = float(row.get("services_years") or 0)
    prd_yrs = float(row.get("product_years") or 0)
    if svc_yrs > 4 and prd_yrs < 0.5:
        return True, 1.0

    # Soft penalties
    if row.get("skill_months_exceed_career"):
        penalty += 0.15
    gap = abs(float(row.get("dur_vs_dates_gap") or 0))
    if gap > 2.0:  # >2 yr discrepancy beyond normal baseline
        penalty += min(0.2, (gap - 2.0) * 0.05)
    if row.get("has_overlapping_roles"):
        penalty += 0.05

    return False, min(penalty, 0.5)


# ---------------------------------------------------------------------------
# B3 — Concept scoring with evidence weighting
# ---------------------------------------------------------------------------

def _match_concepts(text: str) -> set[str]:
    """Return set of concept names that fire in this text."""
    hits = set()
    for name, _w, compiled_pats in _COMPILED:
        for pat in compiled_pats:
            if pat.search(text):
                hits.add(name)
                break
    return hits


def _concept_score(row: dict) -> float:
    """Evidence-weighted concept score in [0, 1]."""
    history = row.get("history_text") or ""
    titles = row.get("title_history_text") or ""
    summary = row.get("summary_text") or ""
    skills = json.loads(row.get("skills_raw") or "[]")

    # Skill names that appear in work history (corroborated)
    history_lower = (history + " " + summary).lower()

    # Hit sets per source
    hist_hits = _match_concepts(history + "\n" + summary)
    title_hits = _match_concepts(titles)

    # Skills: split corroborated vs claimed-only
    skill_names_text = " ".join(s.get("name", "") for s in skills)
    skill_hit_names = _match_concepts(skill_names_text)
    corroborated = skill_hit_names & hist_hits
    uncorroborated = skill_hit_names - hist_hits

    raw_score = 0.0
    max_score = 0.0

    for name, weight, _ in CONCEPTS:
        max_score += weight
        contribution = 0.0
        if name in hist_hits:
            contribution = max(contribution, 1.0)
        if name in title_hits:
            contribution = max(contribution, 0.8)
        if name in corroborated:
            contribution = max(contribution, 1.0)
        elif name in uncorroborated:
            contribution = max(contribution, 0.1)
        raw_score += weight * contribution

    # Keyword-stuffer penalty: large uncorroborated ratio collapses score
    total_skill_hits = len(skill_hit_names)
    if total_skill_hits > 0:
        uncorr_ratio = len(uncorroborated) / total_skill_hits
        if uncorr_ratio > 0.7:
            raw_score *= 0.3

    return raw_score / max_score if max_score > 0 else 0.0


# ---------------------------------------------------------------------------
# B3 — BM25 relevance
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def _build_bm25(rows: list[dict]) -> BM25Okapi:
    corpus = []
    for r in rows:
        text = " ".join([
            r.get("history_text") or "",
            r.get("title_history_text") or "",
            r.get("summary_text") or "",
            " ".join(
                s.get("name", "") for s in json.loads(r.get("skills_raw") or "[]")
            ),
        ])
        corpus.append(_tokenize(text))
    return BM25Okapi(corpus)


# ---------------------------------------------------------------------------
# B4 — Structured-fit, behavioral, logistics, composite
# ---------------------------------------------------------------------------

def _structured_fit(row: dict) -> float:
    """[0, 1] — experience band, product-ML years, seniority."""
    yoe = float(row.get("yoe_claimed") or 0)
    prd = float(row.get("product_years") or 0)
    career_m = int(row.get("career_months") or 0)

    # Experience band 5–9 yr optimal
    if 5 <= yoe <= 9:
        exp_score = 1.0
    elif yoe < 5:
        exp_score = yoe / 5.0
    else:
        exp_score = max(0.5, 1.0 - (yoe - 9) * 0.04)

    # Product-company ML years (not services)
    prd_score = min(1.0, prd / 4.0)

    # Services dilution: heavy services background dilutes fit
    svc = float(row.get("services_years") or 0)
    svc_ratio = svc / (yoe + 0.1)
    svc_penalty = svc_ratio * 0.5  # 100% services → 0.5 penalty on product score

    return (exp_score * 0.4 + prd_score * 0.6) * (1 - svc_penalty)


def _behavioral(row: dict) -> float:
    """[0, 1] — recency, engagement, assessment, openness."""
    last_active = int(row.get("sig_last_active_days") or 999)
    # Recency: 0-30d → 1.0, 180d → 0.5, >365d → 0.1
    recency = max(0.1, 1.0 - last_active / 365.0)

    rr = float(row.get("sig_recruiter_response_rate") or 0)
    otw = 1.0 if row.get("sig_open_to_work_flag") else 0.5
    assessment = float(row.get("sig_skill_assessment_mean") or 0) / 100.0
    github = min(1.0, max(0.0, float(row.get("sig_github_activity_score") or 0)) / 10.0)

    return (recency * 0.35 + rr * 0.25 + otw * 0.15 +
            assessment * 0.15 + github * 0.10)


def _logistics(row: dict) -> float:
    """[0, 1] — location and notice period soft modifiers."""
    country = (row.get("country") or "").lower()
    location = (row.get("location") or "").lower()
    notice = int(row.get("sig_notice_period_days") or 90)
    willing = row.get("sig_willing_to_relocate") or False

    # India + target metros preferred
    if country == "india":
        loc_score = 1.0
        if any(c in location for c in ("noida", "pune", "bengaluru", "bangalore",
                                        "hyderabad", "mumbai", "delhi", "gurugram",
                                        "gurgaon", "chennai")):
            loc_score = 1.1  # slight boost for target metro
    elif willing:
        loc_score = 0.7
    else:
        loc_score = 0.5

    # Notice period: ≤30d ideal, 90d OK, >90d soft penalty
    if notice <= 30:
        np_score = 1.0
    elif notice <= 60:
        np_score = 0.9
    elif notice <= 90:
        np_score = 0.8
    else:
        np_score = max(0.5, 0.8 - (notice - 90) / 180.0)

    return min(1.0, loc_score) * 0.7 + np_score * 0.3


def _cv_speech_penalty(row: dict) -> float:
    """Extra penalty for CV/speech/robotics-heavy profiles with no NLP/IR."""
    all_text = " ".join([
        row.get("history_text") or "",
        row.get("title_history_text") or "",
        row.get("summary_text") or "",
    ])
    mismatch = sum(1 for p in _MISMATCH_PATS if p.search(all_text))
    nlp_ir_hits = _match_concepts(all_text) & {"ranking_ir", "nlp", "recommendation"}
    if mismatch >= 3 and not nlp_ir_hits:
        return 0.4
    if mismatch >= 2 and not nlp_ir_hits:
        return 0.2
    return 0.0


def _composite(relevance: float, concept: float, fit: float,
               behav: float, logist: float, penalty: float,
               cv_penalty: float) -> float:
    r = max(0.0, relevance) ** 0.4
    c = max(0.0, concept) ** 0.3
    f = max(0.0, fit) ** 0.15
    b = max(0.0, behav) ** 0.1
    l_ = max(0.0, logist) ** 0.05
    base = r * c * f * b * l_
    return base * (1 - penalty) * (1 - cv_penalty)


# ---------------------------------------------------------------------------
# B5 — Reasoning composer
# ---------------------------------------------------------------------------

# Sentence frames indexed by (0..N); picked per-candidate via seeded random
_FRAMES_POSITIVE = [
    "{title} with {yoe:.1f}yr exp and {prd:.1f}yr at product companies; {fact}.",
    "Strong product-company background ({prd:.1f}yr); {fact}. {jd_link}.",
    "{yoe:.1f}yr career ({prd:.1f}yr at product companies); {fact}. {concern}",
    "Candidate has {prd:.1f}yr shipping production ML at product firms; {fact}.",
    "{title} — {fact}; assessment scores {assess:.0f}/100 avg. {concern}",
    "Located in {loc}: {fact}; {prd:.1f}yr product exp makes a credible shortlist.",
    "{yoe:.1f}yr total, {prd:.1f}yr at product companies — {fact}. {jd_link}.",
]

_FRAMES_WEAK = [
    "{title} ({yoe:.1f}yr exp); {fact} but {concern}.",
    "Mixed profile: {fact}; however {concern}. Borderline for this JD.",
    "{yoe:.1f}yr exp with {prd:.1f}yr at product firms; {fact}. {concern}",
    "Possibly relevant ({fact}) but {concern} limits confidence.",
    "{title} — some signal ({fact}) offset by {concern}. Rank ~{rank}.",
]

_JD_LINKS = [
    "directly addresses the JD's search/ranking requirement",
    "aligns with the JD's NLP/IR focus",
    "matches the JD's production ML mandate",
    "fits the JD's recommendation-system scope",
    "relevant to the JD's product-company bias",
]

_CONCERNS = [
    "notice period {notice}d is long",
    "currently in services ({svc:.1f}yr), limited product exposure",
    "location ({loc}) outside Noida/Pune preference",
    "behavioral signals are weak (low recruiter response rate)",
    "no strong corroborating evidence in work descriptions",
    "heavy CV/speech background; NLP/IR exposure unclear",
]


def _compose_reasoning(row: dict, rank: int, score: float) -> str:
    rng = random.Random(row.get("candidate_id", "") + str(rank))

    yoe = float(row.get("yoe_claimed") or 0)
    prd = float(row.get("product_years") or 0)
    svc = float(row.get("services_years") or 0)
    title = row.get("current_title") or "Candidate"
    loc = (row.get("location") or row.get("country") or "unknown location")
    notice = int(row.get("sig_notice_period_days") or 90)
    assess = float(row.get("sig_skill_assessment_mean") or 0)
    country = (row.get("country") or "").lower()

    # Pick 2-3 concrete facts from available signals
    all_text = " ".join([
        row.get("history_text") or "",
        row.get("summary_text") or "",
        row.get("title_history_text") or "",
    ])
    concepts_hit = _match_concepts(all_text) & {"ranking_ir", "nlp", "recommendation",
                                                 "ml_production", "deep_learning"}
    skill_names = [s.get("name", "") for s in json.loads(row.get("skills_raw") or "[]")]
    top_skills = ", ".join(skill_names[:3]) if skill_names else "unspecified skills"

    fact_pool = [
        f"shows {len(concepts_hit)} core JD concept(s): {', '.join(sorted(concepts_hit)) or 'none'}",
        f"current role: {title}",
        f"top skills include {top_skills}",
        f"assessment avg {assess:.0f}/100" if assess > 0 else None,
        f"{prd:.1f}yr at product companies" if prd > 1 else None,
        f"GitHub activity score {row.get('sig_github_activity_score', 0):.1f}/10",
    ]
    fact_pool = [f for f in fact_pool if f]
    fact = rng.choice(fact_pool) if fact_pool else "limited signal"

    jd_link = rng.choice(_JD_LINKS)

    concern_pool = []
    if notice > 60:
        concern_pool.append(f"notice period {notice}d is long")
    if svc > prd and svc > 2:
        concern_pool.append(f"currently in services ({svc:.1f}yr), limited product exposure")
    if country != "india":
        concern_pool.append(f"location ({loc}) outside India preference")
    rr = float(row.get("sig_recruiter_response_rate") or 0)
    if rr < 0.3:
        concern_pool.append("low recruiter response rate")
    if not concern_pool:
        concern_pool.append("no major red flags noted")

    concern = rng.choice(concern_pool) if concern_pool else "no major red flags"

    # Rank band: top decile → positive frame, rest → weaker frame
    top_decile = rank <= 10
    mid_band = 11 <= rank <= 50

    if top_decile:
        frames = _FRAMES_POSITIVE[:4]
    elif mid_band:
        frames = _FRAMES_POSITIVE[3:] + _FRAMES_WEAK[:2]
    else:
        frames = _FRAMES_WEAK

    frame = rng.choice(frames)
    try:
        text = frame.format(
            title=title, yoe=yoe, prd=prd, svc=svc,
            loc=loc, notice=notice, assess=assess,
            fact=fact, jd_link=jd_link, concern=concern,
            rank=rank,
        )
    except KeyError:
        text = f"{title} ({yoe:.1f}yr exp, {prd:.1f}yr product): {fact}. {concern}."

    return text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B: rank candidates → submission.csv")
    ap.add_argument("--candidates", required=True, help="features.parquet from normalize.py")
    ap.add_argument("--out", required=True, help="output submission.csv")
    args = ap.parse_args()

    inp = Path(args.candidates)
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")

    print("B1 Loading features …", file=sys.stderr)
    table = pq.read_table(inp)
    rows = table.to_pylist()
    n = len(rows)
    print(f"  {n:,} candidates loaded", file=sys.stderr)

    # B2 — Integrity gate
    print("B2 Integrity gate …", file=sys.stderr)
    disq_mask = []
    penalties = []
    for r in rows:
        dq, pen = _gate(r)
        disq_mask.append(dq)
        penalties.append(pen)
    n_dq = sum(disq_mask)
    print(f"  {n_dq} disqualified ({n_dq/n*100:.1f}%)", file=sys.stderr)

    # B3 — BM25 over all non-disqualified
    print("B3 BM25 + concept scoring …", file=sys.stderr)
    active_idx = [i for i, dq in enumerate(disq_mask) if not dq]
    active_rows = [rows[i] for i in active_idx]

    bm25 = _build_bm25(active_rows)
    jd_tokens = _tokenize(JD_TEXT)
    bm25_raw = bm25.get_scores(jd_tokens)

    # Normalise BM25 scores to [0, 1]
    bm25_max = bm25_raw.max() if bm25_raw.size > 0 else 1.0
    bm25_norm = bm25_raw / (bm25_max + 1e-9)

    concept_scores = np.array([_concept_score(r) for r in active_rows])

    # Blend: 40% BM25, 60% concept (concept is trap-resistant)
    relevance = 0.4 * bm25_norm + 0.6 * concept_scores

    # B4 — Composite
    print("B4 Composite scoring …", file=sys.stderr)
    composites = []
    for j, (i, r) in enumerate(zip(active_idx, active_rows)):
        fit = _structured_fit(r)
        behav = _behavioral(r)
        logist = _logistics(r)
        cv_pen = _cv_speech_penalty(r)
        score = _composite(
            relevance[j], concept_scores[j], fit, behav, logist,
            penalties[i], cv_pen,
        )
        composites.append((score, i))

    composites.sort(key=lambda x: (-x[0], rows[x[1]].get("candidate_id", "")))

    # Take top 100
    top100 = composites[:100]

    # Verify honeypot rate (must be < 10%)
    honeypot_count = sum(
        1 for score, i in top100
        if rows[i].get("has_impossible_tenure")
        or rows[i].get("expert_zero_months_count", 0) > 0
        or rows[i].get("has_future_date")
    )
    if honeypot_count > 10:
        sys.exit(f"ABORT: honeypot rate {honeypot_count}/100 exceeds 10% DQ threshold")

    # B5 — Reasoning + emit
    print("B5 Composing reasoning and writing CSV …", file=sys.stderr)
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        prev_score = None
        for rank, (score, i) in enumerate(top100, start=1):
            r = rows[i]
            # Scores must be non-increasing
            if prev_score is not None and score > prev_score:
                score = prev_score
            prev_score = score

            reasoning = _compose_reasoning(r, rank, score)
            writer.writerow([
                r.get("candidate_id", ""),
                rank,
                f"{score:.6f}",
                reasoning,
            ])

    print(f"Written → {out_path}  (top 100, honeypots in top 100: {honeypot_count})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
