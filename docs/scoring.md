# Scoring

## Composite — multiplicative gating, not a weighted sum

```
score = relevance^a · structured_fit^b · behavioral^c · logistics^d · (1 - penalty)
```

Multiplicative because one terrible dimension must NOT be averaged away by keyword
brilliance — that is exactly what the traps probe. Weights live in a committed YAML
config so git history shows the tuning.

- **relevance** (B3): dense cosine + evidence-weighted concept score. See below.
- **structured_fit**: 5–9 yr band (soft), applied-ML years at *product* companies
  (`product_years` vs `services_years`), evidence of shipping search/ranking/reco.
- **behavioral**: recency (`last_active_date`), `recruiter_response_rate`,
  `open_to_work_flag`, assessment scores. The JD says down-weight the perfect-on-paper
  ghost; behavioral twins exist to verify this term carries real weight.
- **logistics**: location (India / target metros) and `notice_period_days` as soft
  modifiers, not hard gates.

## Evidence-weighted skills (the trap-killer)

Every concept hit is tagged by source field; source sets its multiplier:

| Source | Multiplier |
|---|---|
| Work history / project description | 1.0 |
| Job title | 0.8 |
| Skills list, corroborated by a work-history hit | 1.0 |
| Skills list, uncorroborated | 0.1 |

Per-concept contributions are capped (diminishing returns). Apply a stuffing penalty
when claimed-but-unevidenced concept ratio is extreme. Net effect: the keyword stuffer
(many claimed, ~0 evidenced) collapses to near-zero; the plain-language strong
candidate (evidenced phrases like "recommendation engine", "search relevance") scores
high without any JD buzzwords. Same mechanism, opposite outcomes.

## Why ontology AND embeddings (redundant on purpose)

Ontology = precise, explainable per-concept signals. Embeddings = recall on phrasings
no pattern list anticipated. Keep both.