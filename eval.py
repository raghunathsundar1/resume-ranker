#!/usr/bin/env python3
"""Local evaluation: NDCG@10, NDCG@50, P@5, P@10, AP on the 50-candidate sample.

Run:
    python eval.py

Requires: normalize.py, rank.py, data/sample_candidates.json
Keeps labels fixed; cite the delta in commit messages when a change moves them.
"""

import csv
import math
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hand-labeled relevance tiers
# Rubric (JD: Senior ML Engineer — Search / Ranking / Recommendation):
#   3 = highly relevant: direct search/ranking/reco at product co, ML in production
#   2 = relevant: ML/DS background with evidenced adjacent skills at product co
#   1 = tangential: technical background but no ML/search focus, or weak evidence
#   0 = not relevant: non-ML roles, honeypots, JD disqualifiers
# Labels are FIXED — do not adjust to flatter current scores.
# ---------------------------------------------------------------------------
LABELS: dict[str, int] = {
    "CAND_0000001": 1,  # Backend Eng @ Mindtree (services/data pipelines); NLP skills uncorroborated
    "CAND_0000002": 0,  # Operations Manager, IT Services, no ML
    "CAND_0000003": 0,  # Customer Support, 1yr, no ML
    "CAND_0000004": 0,  # Marketing Manager, Paper Products
    "CAND_0000005": 0,  # Accountant, Manufacturing
    "CAND_0000006": 0,  # Business Analyst, Conglomerate, no ML
    "CAND_0000007": 0,  # Civil Engineer, IT Services
    "CAND_0000008": 0,  # Operations Manager, IT Services
    "CAND_0000009": 0,  # Mechanical Engineer, Paper Products
    "CAND_0000010": 1,  # Data Engineer @ Ola (product/transport), Kubeflow adjacent; no ML production
    "CAND_0000011": 0,  # QA Engineer, reco claim in skills only — uncorroborated
    "CAND_0000012": 0,  # Operations Manager, Manufacturing, 1yr
    "CAND_0000013": 0,  # Civil Engineer, Manufacturing, 1yr
    "CAND_0000014": 1,  # Frontend Eng @ Zomato (product); FAISS in skills but work history is frontend
    "CAND_0000015": 2,  # SWE @ Razorpay (fintech product); PyTorch + W&B evidenced in work history
    "CAND_0000016": 0,  # Accountant, IT Services
    "CAND_0000017": 0,  # Accountant, IT Services
    "CAND_0000018": 0,  # Frontend Eng, Manufacturing; CNN claim uncorroborated
    "CAND_0000019": 0,  # Project Manager, Conglomerate
    "CAND_0000020": 0,  # Mechanical Engineer, IT Services
    "CAND_0000021": 0,  # Project Manager, IT Services
    "CAND_0000022": 0,  # Mechanical Engineer, 1yr
    "CAND_0000023": 0,  # SWE, Manufacturing, no ML evidence
    "CAND_0000024": 0,  # HR Manager, IT Services
    "CAND_0000025": 0,  # Frontend Eng, IT Services
    "CAND_0000026": 0,  # Graphic Designer; Kubeflow/Beam claims uncorroborated — stuffer
    "CAND_0000027": 0,  # DevOps Eng, IT Services; YOLO/PEFT claims not in work history
    "CAND_0000028": 0,  # Operations Manager, IT Services, 1yr
    "CAND_0000029": 0,  # Business Analyst, IT Services
    "CAND_0000030": 0,  # Marketing Manager, Paper Products — keyword stuffer
    "CAND_0000031": 3,  # Reco Systems Eng @ Swiggy + Search Eng @ Mad Street Den — direct match
    "CAND_0000032": 0,  # .NET Developer, IT Services; speech claim in skills only
    "CAND_0000033": 0,  # Graphic Designer, IT Services
    "CAND_0000034": 0,  # Business Analyst, IT Services, 14yr consulting
    "CAND_0000035": 1,  # Full Stack Dev, Manufacturing; reco in skills but work is frontend/fullstack
    "CAND_0000036": 0,  # Project Manager, Software
    "CAND_0000037": 0,  # Business Analyst, Manufacturing
    "CAND_0000038": 1,  # Java Dev @ Swiggy (product); Kubeflow adjacent, not ML production
    "CAND_0000039": 0,  # Marketing Manager, Manufacturing
    "CAND_0000040": 0,  # Customer Support, Manufacturing, 1yr
    "CAND_0000041": 0,  # Operations Manager, Software
    "CAND_0000042": 0,  # HR Manager, Conglomerate
    "CAND_0000043": 2,  # Cloud Eng @ Swiggy (product); Elasticsearch/OpenSearch — search infra
    "CAND_0000044": 0,  # Frontend Eng, IT Services
    "CAND_0000045": 0,  # Project Manager, Software
    "CAND_0000046": 0,  # Mechanical Engineer, Software
    "CAND_0000047": 0,  # Project Manager, IT Services, 2yr
    "CAND_0000048": 1,  # Mobile Dev @ CRED (fintech product); not ML, product-co experience
    "CAND_0000049": 0,  # Mechanical Engineer, Conglomerate
    "CAND_0000050": 0,  # Business Analyst, IT Services, 13yr consulting
}

_TIER_NAMES = ["not-relevant", "tangential", "relevant", "highly-relevant"]


# ---------------------------------------------------------------------------
# Metrics (hand-rolled so the logic is interview-defensible)
# ---------------------------------------------------------------------------

def _dcg(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_ids: list[str], labels: dict[str, int], k: int) -> float:
    actual = [labels.get(cid, 0) for cid in ranked_ids]
    ideal = sorted(labels.get(cid, 0) for cid in ranked_ids)[::-1]
    idcg = _dcg(ideal, k)
    return _dcg(actual, k) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids: list[str], labels: dict[str, int],
                   k: int, threshold: int = 1) -> float:
    return sum(1 for cid in ranked_ids[:k] if labels.get(cid, 0) >= threshold) / k


def average_precision(ranked_ids: list[str], labels: dict[str, int],
                      threshold: int = 1) -> float:
    relevant = [cid for cid in ranked_ids if labels.get(cid, 0) >= threshold]
    if not relevant:
        return 0.0
    hits, precs = 0, []
    for i, cid in enumerate(ranked_ids):
        if labels.get(cid, 0) >= threshold:
            hits += 1
            precs.append(hits / (i + 1))
    return sum(precs) / len(relevant)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline() -> list[str]:
    print("Running normalize.py on sample …")
    r1 = subprocess.run(
        [sys.executable, "normalize.py",
         "--input", "data/sample_candidates.json",
         "--out", "eval_sample.parquet",
         "--json-array"],
        capture_output=True, text=True,
    )
    if r1.returncode != 0:
        sys.exit(f"normalize.py failed:\n{r1.stderr}")

    print("Running rank.py on sample …")
    r2 = subprocess.run(
        [sys.executable, "rank.py",
         "--candidates", "eval_sample.parquet",
         "--out", "eval_submission.csv"],
        capture_output=True, text=True,
    )
    if r2.returncode != 0:
        sys.exit(f"rank.py failed:\n{r2.stderr}")

    with open("eval_submission.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["candidate_id"] for r in sorted(rows, key=lambda r: int(r["rank"]))]


def main() -> None:
    ranked_ids = run_pipeline()

    # Candidates gated out are appended at the end (worst position)
    missing = [cid for cid in LABELS if cid not in ranked_ids]
    full_ranking = ranked_ids + missing

    print(f"\nRanked {len(ranked_ids)} / {len(LABELS)} candidates "
          f"({len(missing)} gated out, appended at tail)\n")

    print("Top-15 ranking:")
    print(f"  {'Rank':>4}  {'CandidateID':<18}  {'Label':>5}  Tier")
    print("  " + "-" * 50)
    for i, cid in enumerate(full_ranking[:15], 1):
        label = LABELS.get(cid, 0)
        marker = " [+]" if label >= 2 else ("" if label == 1 else " [-]")
        print(f"  {i:4d}  {cid:<18}  {label:>5}  {_TIER_NAMES[label]}{marker}")

    n10 = ndcg_at_k(full_ranking, LABELS, 10)
    n50 = ndcg_at_k(full_ranking, LABELS, 50)
    p5  = precision_at_k(full_ranking, LABELS, 5)
    p10 = precision_at_k(full_ranking, LABELS, 10)
    ap  = average_precision(full_ranking, LABELS)

    print("\n" + "-" * 40)
    print(f"  NDCG@10  : {n10:.4f}")
    print(f"  NDCG@50  : {n50:.4f}")
    print(f"  P@5      : {p5:.4f}")
    print(f"  P@10     : {p10:.4f}")
    print(f"  AP       : {ap:.4f}")
    print("-" * 40)

    # Non-relevant rate in the ranking (proxy for honeypot check on sample)
    non_rel = sum(1 for cid in ranked_ids if LABELS.get(cid, 0) == 0)
    print(f"\n  Non-relevant in ranking: {non_rel}/{len(ranked_ids)} "
          f"= {non_rel/len(ranked_ids)*100:.1f}%")
    if non_rel / len(ranked_ids) > 0.10:
        print("  WARNING: non-relevant rate > 10%")


if __name__ == "__main__":
    main()
