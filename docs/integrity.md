# Integrity gate (B2)

Runs vectorized over all 100K. Produces a `disqualified` mask (hard exclusion) and a
`penalty` vector (soft caps). `normalize.py` only EXPOSES raw numbers; the gate owns
all thresholds, in one auditable place.

## Honeypot signals (~80 exist in the pool; >10% in top 100 = DQ)

Hard, unambiguous (→ exclude):
- `impossible_tenure` — one role longer than the whole claimed career.
- `expert_zero_months_count` > 0 — "expert" proficiency with ~0 months of use.
- `has_future_date` — start in the future, or end before start.
- Company-age contradiction — tenure at a company exceeding the company's age.
  **Cannot be computed in `normalize.py`** (records lack founding dates); needs the
  A3 company table. Implement here.

Soft (→ penalty, require corroboration, do NOT hard-exclude):
- `skill_months_exceed_career` — fires ~9/50 on the clean sample, mostly legitimate
  (skills span overlapping roles). Treat as weak signal only.
- `dur_vs_dates_gap` — `duration_months` disagreeing with parsed dates. Note there is
  a normal ~+0.3–0.5y baseline for everyone (see gotchas.md); flag deviation, not presence.
- `has_overlapping_roles` — already tolerance-gated to >6 cumulative months in A1.

## JD disqualifiers (hard caps / zeroing, per the JD's own language)

- Consulting/services-only career (`services_years` ≫ `product_years`, no product roles).
- CV/speech/robotics-only with no NLP/IR exposure.
- Pure-research career with no production deployment.
- AI experience that is only sub-12-month LLM-framework glue (LangChain-calling-OpenAI).
- 18+ months with no individual-contributor coding role.

Per the spec: a good system avoids honeypots *naturally* by reading profiles; you
should not need to special-case them. The explicit checks are cheap insurance given
the DQ rule, not the primary mechanism.