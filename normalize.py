#!/usr/bin/env python3
"""Phase A1: raw JSONL / JSON-array → features.parquet.

Field-name mappings live ONLY in the *_MAP constants at the top.
Edit those tables (and nothing else) when the full pool reveals alias names.

Usage:
    python normalize.py --input candidates.jsonl.gz --out features.parquet
    python normalize.py --input sample_candidates.json --out sample.parquet --json-array
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from dateutil import parser as du_parser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AS_OF = date(2026, 1, 31)   # pinned; never replace with datetime.now()

# ---------------------------------------------------------------------------
# Field-name maps — raw JSON key → canonical name.
# ---------------------------------------------------------------------------

PROFILE_MAP: dict[str, str] = {
    "anonymized_name": "name",
    "headline": "headline",
    "summary": "summary",
    "location": "location",
    "country": "country",
    "years_of_experience": "yoe_claimed",
    "current_title": "current_title",
    "current_company": "current_company",
    "current_company_size": "current_company_size",
    "current_industry": "current_industry",
}

EMP_MAP: dict[str, str] = {
    "company": "company",
    "title": "title",
    "start_date": "start_date",
    "end_date": "end_date",
    "duration_months": "duration_months",
    "is_current": "is_current",
    "industry": "industry",
    "company_size": "company_size",
    "description": "description",
}

SKILL_MAP: dict[str, str] = {
    "name": "name",
    "proficiency": "proficiency",
    "endorsements": "endorsements",
    "duration_months": "duration_months",
}

# ---------------------------------------------------------------------------
# Industry taxonomy
# ---------------------------------------------------------------------------

_SERVICES_TOKENS: frozenset[str] = frozenset({
    "IT Services", "Information Technology & Services",
    "Information Technology and Services",
    "Consulting", "Management Consulting", "Business Consulting",
    "Technology Consulting", "IT Consulting",
    "Staffing & Recruiting", "Outsourcing/Offshoring",
    "BPO", "Professional Services", "Systems Integration",
    "Technology Services", "Computer & Network Security",
})

_PRODUCT_TOKENS: frozenset[str] = frozenset({
    "Internet", "Computer Software", "Software", "SaaS",
    "E-Commerce", "E-commerce", "Technology", "Semiconductors",
    "Consumer Electronics", "Financial Services", "FinTech",
    "Ed-Tech", "HealthTech", "Health Tech",
    "Media", "Entertainment", "Gaming",
    "Retail Technology", "Mobile", "Telecommunications",
    "Automotive", "Aerospace",
    # Product-oriented tech verticals
    "Food Delivery", "Food Technology", "FoodTech",
    "AI", "AI/ML", "Artificial Intelligence", "Machine Learning",
    "Cybersecurity", "AdTech", "HRTech", "LegalTech",
    "PropTech", "InsurTech", "CleanTech", "AgriTech",
    "Logistics Technology", "Transport Technology",
})

_RESEARCH_TOKENS: frozenset[str] = frozenset({
    "Research", "Higher Education", "Education", "Academic",
    "University", "Institute", "Non-profit", "Government",
    "Defense & Space", "Think Tank", "NGO",
})


def _classify_industry(ind: str) -> str:
    """'services' | 'product' | 'research' | 'other'"""
    if not ind:
        return "other"
    if ind in _SERVICES_TOKENS:
        return "services"
    il = ind.lower()
    for t in _SERVICES_TOKENS:
        if len(t) > 6 and t.lower() in il:
            return "services"
    if ind in _PRODUCT_TOKENS:
        return "product"
    for t in _PRODUCT_TOKENS:
        if len(t) > 6 and t.lower() in il:
            return "product"
    if ind in _RESEARCH_TOKENS:
        return "research"
    return "other"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return du_parser.parse(s).date()
    except Exception:
        return None


def _date_months(d: date) -> int:
    return d.year * 12 + d.month


def _role_date_months(start: date | None, end: date | None) -> int:
    if start is None:
        return 0
    e = end or AS_OF
    if e < start:
        return 0
    return _date_months(e) - _date_months(start)


# ---------------------------------------------------------------------------
# Title normalizer
# ---------------------------------------------------------------------------

_ABBREV_RE = re.compile(r"[.\-/]")
_MULTI_SPACE_RE = re.compile(r"\s+")

def _norm_title(t: str) -> str:
    t = _ABBREV_RE.sub(" ", t or "")
    return _MULTI_SPACE_RE.sub(" ", t).strip()


# ---------------------------------------------------------------------------
# Career feature extraction
# ---------------------------------------------------------------------------

def _process_career(career_history: list[dict]) -> dict[str, Any]:
    """Extract career stats and integrity signals from the job history list."""
    career_months = 0
    services_months = 0
    product_months = 0
    research_months = 0
    current_role_months = 0
    date_tenure_months = 0

    title_parts: list[str] = []
    desc_parts: list[str] = []

    # For overlap detection: collect (start_date, end_date) pairs
    intervals: list[tuple[date, date]] = []

    has_future_date = False
    has_impossible_single = False   # set later after summing career_months

    roles_with_months: list[int] = []

    for raw in (career_history or []):
        r = {EMP_MAP.get(k, k): v for k, v in raw.items()}
        dm = int(r.get("duration_months") or 0)
        career_months += dm
        roles_with_months.append(dm)

        ind = r.get("industry", "") or ""
        cat = _classify_industry(ind)
        if cat == "services":
            services_months += dm
        elif cat == "product":
            product_months += dm
        elif cat == "research":
            research_months += dm

        if r.get("is_current"):
            current_role_months = dm

        title = _norm_title(r.get("title") or "")
        if title:
            title_parts.append(title)
        desc = (r.get("description") or "").strip()
        if desc:
            desc_parts.append(desc)

        # Date parsing for cross-checks
        start = _parse_date(r.get("start_date"))
        end_raw = _parse_date(r.get("end_date"))
        end = end_raw or (AS_OF if r.get("is_current") else None)

        if start and start > AS_OF:
            has_future_date = True
        if end_raw and start and end_raw < start:
            has_future_date = True

        if start and end and end >= start:
            intervals.append((start, end))
            date_tenure_months += _role_date_months(start, end)

    # Impossible single tenure: one role longer than total claimed career
    if career_months > 0:
        has_impossible_single = any(dm > career_months for dm in roles_with_months
                                    if len(roles_with_months) > 1)

    # Overlap detection: cumulative overlap > 6 months
    total_overlap = _compute_overlap_months(intervals)
    has_overlapping_roles = total_overlap > 6.0

    # dur_vs_dates_gap in years: positive = duration_months overstates dates
    dur_vs_dates_gap = (career_months - date_tenure_months) / 12.0

    return {
        "career_months": career_months,
        "services_months": services_months,
        "product_months": product_months,
        "research_months": research_months,
        "services_years": round(services_months / 12.0, 2),
        "product_years": round(product_months / 12.0, 2),
        "job_count": len(career_history or []),
        "current_role_months": current_role_months,
        "title_history_text": " | ".join(title_parts),
        "history_text": "\n\n".join(desc_parts),
        "has_impossible_tenure": has_impossible_single,
        "has_future_date": has_future_date,
        "dur_vs_dates_gap": round(dur_vs_dates_gap, 3),
        "has_overlapping_roles": has_overlapping_roles,
    }


def _compute_overlap_months(intervals: list[tuple[date, date]]) -> float:
    """Total months of pairwise concurrent employment."""
    if len(intervals) < 2:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    for i in range(len(intervals) - 1):
        a_s, a_e = intervals[i]
        for j in range(i + 1, len(intervals)):
            b_s, b_e = intervals[j]
            if b_s >= a_e:
                break  # sorted, no further overlap with i
            overlap_end = min(a_e, b_e)
            overlap_start = max(a_s, b_s)
            if overlap_end > overlap_start:
                total += (overlap_end - overlap_start).days / 30.44
    return total


# ---------------------------------------------------------------------------
# Skills feature extraction
# ---------------------------------------------------------------------------

def _process_skills(skills: list[dict], career_months: int) -> dict[str, Any]:
    normalized: list[dict] = []
    expert_zero_count = 0
    advanced_count = 0
    expert_count = 0
    total_skill_months = 0

    for raw in (skills or []):
        s = {SKILL_MAP.get(k, k): v for k, v in raw.items()}
        prof = (s.get("proficiency") or "").lower()
        dm = int(s.get("duration_months") or 0)
        total_skill_months += dm

        if prof == "expert" and dm <= 1:
            expert_zero_count += 1
        if prof in ("advanced", "expert"):
            advanced_count += 1
        if prof == "expert":
            expert_count += 1

        normalized.append(s)

    # Fires when total claimed skill-months is implausibly high vs career length.
    # Threshold = 7× allows legitimate multi-role skill overlap without false positives.
    # Expect ~9/50 on the clean sample (soft signal only; see integrity.md).
    skill_months_exceed = (
        career_months > 0 and total_skill_months > career_months * 7
    )

    return {
        "skill_count": len(normalized),
        "advanced_skill_count": advanced_count,
        "expert_skill_count": expert_count,
        "expert_zero_months_count": expert_zero_count,
        "skill_months_exceed_career": skill_months_exceed,
        "skills_raw": json.dumps(normalized, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Education feature extraction
# ---------------------------------------------------------------------------

_TIER_ORDER = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "tier_5": 5}
_MS_RE = re.compile(r"\b(M\.?S\.?|M\.?E\.?|M\.?Tech|Master|MTech|MSc)\b", re.I)
_PHD_RE = re.compile(r"\b(Ph\.?D\.?|Doctor|D\.Sc)\b", re.I)
_CS_RE = re.compile(
    r"\b(Computer|Software|Comp Sci|CS|Data Sci|Information Tech|AI|ML|"
    r"Electronics|Electrical|Math|Statistics|Physics)\b", re.I
)


def _process_education(education: list[dict]) -> dict[str, Any]:
    best_tier = "tier_5"
    has_ms = False
    has_phd = False
    cs_field = False

    for edu in (education or []):
        tier = edu.get("tier", "tier_5")
        if _TIER_ORDER.get(tier, 5) < _TIER_ORDER.get(best_tier, 5):
            best_tier = tier
        deg = edu.get("degree", "") or ""
        if _PHD_RE.search(deg):
            has_phd = True
        elif _MS_RE.search(deg):
            has_ms = True
        fos = edu.get("field_of_study", "") or ""
        if _CS_RE.search(fos) or _CS_RE.search(deg):
            cs_field = True

    return {
        "edu_tier_best": best_tier,
        "edu_has_ms": has_ms,
        "edu_has_phd": has_phd,
        "edu_cs_field": cs_field,
    }


# ---------------------------------------------------------------------------
# redrob_signals flattener
# ---------------------------------------------------------------------------

def _process_signals(sigs: dict) -> dict[str, Any]:
    if not sigs:
        sigs = {}

    # Parse last_active_date to days before AS_OF
    lad = _parse_date(sigs.get("last_active_date"))
    last_active_days = (AS_OF - lad).days if lad else 999

    salary_range = sigs.get("expected_salary_range_inr_lpa") or {}

    scores_dict = sigs.get("skill_assessment_scores") or {}
    scores_vals = [float(v) for v in scores_dict.values() if v is not None]
    sa_mean = round(sum(scores_vals) / len(scores_vals), 2) if scores_vals else 0.0
    sa_max = round(max(scores_vals), 2) if scores_vals else 0.0

    return {
        "sig_profile_completeness_score": float(sigs.get("profile_completeness_score") or 0),
        "sig_last_active_days": last_active_days,
        "sig_open_to_work_flag": bool(sigs.get("open_to_work_flag", False)),
        "sig_profile_views_received_30d": int(sigs.get("profile_views_received_30d") or 0),
        "sig_applications_submitted_30d": int(sigs.get("applications_submitted_30d") or 0),
        "sig_recruiter_response_rate": float(sigs.get("recruiter_response_rate") or 0),
        "sig_avg_response_time_hours": float(sigs.get("avg_response_time_hours") or 0),
        "sig_connection_count": int(sigs.get("connection_count") or 0),
        "sig_endorsements_received": int(sigs.get("endorsements_received") or 0),
        "sig_notice_period_days": int(sigs.get("notice_period_days") or 0),
        "sig_salary_min": float(salary_range.get("min") or 0),
        "sig_salary_max": float(salary_range.get("max") or 0),
        "sig_preferred_work_mode": str(sigs.get("preferred_work_mode") or ""),
        "sig_willing_to_relocate": bool(sigs.get("willing_to_relocate", False)),
        "sig_github_activity_score": float(sigs.get("github_activity_score") or 0),
        "sig_search_appearance_30d": int(sigs.get("search_appearance_30d") or 0),
        "sig_saved_by_recruiters_30d": int(sigs.get("saved_by_recruiters_30d") or 0),
        "sig_interview_completion_rate": float(sigs.get("interview_completion_rate") or 0),
        "sig_offer_acceptance_rate": float(sigs.get("offer_acceptance_rate") or 0),
        "sig_verified_email": bool(sigs.get("verified_email", False)),
        "sig_verified_phone": bool(sigs.get("verified_phone", False)),
        "sig_linkedin_connected": bool(sigs.get("linkedin_connected", False)),
        "sig_skill_assessment_mean": sa_mean,
        "sig_skill_assessment_max": sa_max,
        "sig_skill_assessment_json": json.dumps(scores_dict, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Main record processor
# ---------------------------------------------------------------------------

def process_record(raw: dict) -> dict[str, Any]:
    cid = raw.get("candidate_id", "")
    profile_raw = raw.get("profile") or {}
    profile = {PROFILE_MAP.get(k, k): v for k, v in profile_raw.items()}

    career_feats = _process_career(raw.get("career_history") or [])
    career_months = career_feats["career_months"]

    skill_feats = _process_skills(raw.get("skills") or [], career_months)
    edu_feats = _process_education(raw.get("education") or [])
    sig_feats = _process_signals(raw.get("redrob_signals") or {})

    current_ind = profile.get("current_industry", "") or ""
    is_services_now = _classify_industry(current_ind) == "services"

    row: dict[str, Any] = {
        "candidate_id": cid,
        "name": profile.get("name", ""),
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "yoe_claimed": float(profile.get("yoe_claimed") or 0),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "current_company_size": profile.get("current_company_size", ""),
        "current_industry": current_ind,
        "is_services_now": is_services_now,
        "summary_text": profile.get("summary", ""),
    }
    row.update(career_feats)
    row.update(skill_feats)
    row.update(edu_feats)
    row.update(sig_feats)
    return row


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------

def iter_jsonl_gz(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def iter_json_array(path: Path):
    with open(path, "rb") as fh:
        data = orjson.loads(fh.read())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data)}")
    yield from data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A1: raw candidates → features.parquet")
    ap.add_argument("--input", required=True, help="candidates.jsonl.gz or .jsonl or .json")
    ap.add_argument("--out", required=True, help="output .parquet path")
    ap.add_argument("--json-array", action="store_true",
                    help="input is a JSON array (not JSONL)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")

    reader = iter_json_array(inp) if args.json_array else iter_jsonl_gz(inp)

    rows: list[dict] = []
    for i, raw in enumerate(reader):
        row = process_record(raw)
        rows.append(row)
        if (i + 1) % 10_000 == 0:
            print(f"  processed {i + 1:,} records …", file=sys.stderr)

    print(f"Total records: {len(rows):,}", file=sys.stderr)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.out, compression="snappy")
    print(f"Written → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
