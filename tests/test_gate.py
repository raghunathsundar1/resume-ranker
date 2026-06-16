"""Unit tests for the B2 integrity gate.

Honeypots must flag; clean ML candidates must not.
Run: pytest tests/test_gate.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import rank as rk  # noqa: E402  (rank.py is at project root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(**kwargs) -> dict:
    """Minimal clean ML candidate row; override fields to inject defects."""
    base = {
        # Integrity signals
        "has_impossible_tenure": False,
        "expert_zero_months_count": 0,
        "has_future_date": False,
        # Career shape
        "current_title": "Machine Learning Engineer",
        "services_years": 1.0,
        "product_years": 5.0,
        "career_months": 84.0,
        # Soft-penalty signals
        "skill_months_exceed_career": False,
        "dur_vs_dates_gap": 0.5,
        "has_overlapping_roles": False,
        # Pre-computed text (populated by _precompute in real flow)
        "_hist_lower": "built recommendation engine using collaborative filtering",
        "_title_lower": "machine learning engineer",
        "_skill_lower": "pytorch ranking bm25",
        "_tok_hist": rk._tokenize("built recommendation engine using collaborative filtering"),
        "_tok_title": rk._tokenize("machine learning engineer"),
        "_tok_skill": rk._tokenize("pytorch ranking bm25"),
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Hard-disqualify honeypot tests
# ---------------------------------------------------------------------------

def test_honeypot_impossible_tenure():
    dq, _ = rk._gate(_row(has_impossible_tenure=True))
    assert dq, "impossible_tenure must disqualify"


def test_honeypot_expert_zero_months():
    dq, _ = rk._gate(_row(expert_zero_months_count=2))
    assert dq, "expert_zero_months_count > 0 must disqualify"


def test_honeypot_future_date():
    dq, _ = rk._gate(_row(has_future_date=True))
    assert dq, "has_future_date must disqualify"


# ---------------------------------------------------------------------------
# JD-disqualifier tests
# ---------------------------------------------------------------------------

def test_services_only_career():
    dq, _ = rk._gate(_row(services_years=8.0, product_years=0.0))
    assert dq, "services_only (svc>4, prd<0.5) must disqualify"


def test_services_career_with_product_passes():
    dq, _ = rk._gate(_row(services_years=5.0, product_years=2.0))
    assert not dq, "services career with 2yr product should pass"


def test_non_ml_title_no_evidence_disqualified():
    """Marketing Manager with no ML work history must be disqualified."""
    row = _row(
        current_title="Marketing Manager",
        _hist_lower="managed campaigns and brand strategy",
        _title_lower="marketing manager",
        _skill_lower="photoshop excel powerpoint",
        _tok_hist=rk._tokenize("managed campaigns and brand strategy"),
        _tok_title=rk._tokenize("marketing manager"),
        _tok_skill=rk._tokenize("photoshop excel powerpoint"),
    )
    dq, _ = rk._gate(row)
    assert dq, "non-ML title with no ML evidence must disqualify"


def test_non_ml_title_with_strong_ml_evidence_passes():
    """Project Manager who built ranking + recommendation systems should pass."""
    ml_hist = ("built learn to rank system improving ndcg by 0.12 "
               "shipped recommendation engine collaborative filtering")
    row = _row(
        current_title="Project Manager",
        _hist_lower=ml_hist,
        _title_lower="project manager",
        _skill_lower="ranking bm25 recommendation",
        _tok_hist=rk._tokenize(ml_hist),
        _tok_title=rk._tokenize("project manager"),
        _tok_skill=rk._tokenize("ranking bm25 recommendation"),
        services_years=0.0,
        product_years=6.0,
    )
    dq, _ = rk._gate(row)
    assert not dq, "non-ML title with ≥2 strong ML concept hits should pass"


# ---------------------------------------------------------------------------
# Clean-candidate pass-through tests
# ---------------------------------------------------------------------------

def test_clean_ml_candidate_passes():
    dq, penalty = rk._gate(_row())
    assert not dq
    assert penalty == 0.0


def test_borderline_services_just_below_threshold():
    # svc=4.0 is not > 4 — should pass
    dq, _ = rk._gate(_row(services_years=4.0, product_years=0.3))
    assert not dq, "svc=4.0 is not >4; should not disqualify"


# ---------------------------------------------------------------------------
# Soft-penalty tests (should NOT disqualify but should add penalty)
# ---------------------------------------------------------------------------

def test_soft_penalty_skill_months_exceed_career():
    dq, penalty = rk._gate(_row(skill_months_exceed_career=True))
    assert not dq, "skill_months_exceed_career is soft — must not disqualify"
    assert abs(penalty - 0.15) < 1e-9


def test_soft_penalty_overlapping_roles():
    dq, penalty = rk._gate(_row(has_overlapping_roles=True))
    assert not dq, "has_overlapping_roles is soft — must not disqualify"
    assert abs(penalty - 0.05) < 1e-9


def test_soft_penalty_dur_gap():
    # gap = 3.0 → penalty = min(0.2, (3.0 - 2.0) * 0.05) = 0.05
    dq, penalty = rk._gate(_row(dur_vs_dates_gap=3.0))
    assert not dq
    assert penalty > 0


def test_combined_soft_penalties_capped_at_0_5():
    # Pile on all three soft signals; combined should be ≤ 0.5
    dq, penalty = rk._gate(_row(
        skill_months_exceed_career=True,
        has_overlapping_roles=True,
        dur_vs_dates_gap=20.0,
    ))
    assert not dq
    assert penalty <= 0.5
