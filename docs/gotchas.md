# Gotchas (found by testing — do not reintroduce)

Read this before touching parsing or the integrity thresholds.

- **`duration_months` is authoritative for tenure**, not the parsed dates. Dates are
  an independent cross-check; their disagreement (`dur_vs_dates_gap`) is itself a
  signal, not the source of truth.
- **There is a normal baseline offset.** `experience_discrepancy` and
  `dur_vs_dates_gap` sit at ~+0.3–0.5y for *everyone*, because open-ended current
  roles anchor to `AS_OF` while people report total experience conservatively. Flag
  deviation from this baseline, NOT its mere presence.
- **Overlap detection needs tolerance.** Naive interval overlap flagged 60% of the
  sample (normal job transitions / month-rounding). A1 requires >6 cumulative months
  of concurrency to set `has_overlapping_roles`; it then drops to 0/50 — correct.
- **`skill_months_exceed_career` is SOFT** (~9/50 fire, mostly legitimate). Never
  hard-exclude on it.
- **Title normalizer must strip abbreviation dots** ("Sr." left a stray "."). The
  `[.\-/]` strip in `norm_title` handles this — keep it.
- **Company-age honeypot is not detectable in `normalize.py`** — records lack company
  founding dates. It belongs in B2 with the A3 company table.
- **The sample is clean by design.** ~80 honeypots in 100K ⇒ ~0 in any 50-record
  sample. Do not loosen the gate because it rarely fires on the sample.

## AS_OF

`AS_OF = date(2026, 1, 31)`, just past the latest `last_active_date` in the pool.
Used for all open-ended tenure math. Pinned for reproducibility — never replace with
the current date.