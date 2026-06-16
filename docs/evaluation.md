# Local evaluation (there is no leaderboard)

You get zero feedback until submissions close, so you cannot tune against the real
score. Build your own ground truth and treat it as fixed.

- Hand-label 150–250 candidates into relevance tiers using a written rubric derived
  from the JD. Deliberately include hard cases: spotted keyword stuffers,
  plain-language strong profiles, candidates with bad behavioral signals.
- Keep the label set FIXED across iterations so deltas are comparable.
- Compute NDCG@10, NDCG@50, MAP, P@10 on every change. Use `sklearn.ndcg_score` or a
  hand-rolled version (writing it yourself means you can explain it in the interview).
- Cite the metric delta in the commit message when a change moves it.

## Gate unit tests

Separately from ranking metrics, unit-test the integrity gate (pytest): synthetic
honeypots must flag; clean profiles must not. This doubles as Stage-4 evidence of real
engineering and guards against regressions while tuning.