# Reasoning composer (B5)

Stage 4 samples 10 random rows of the submission and scores each `reasoning` string.
The composer must satisfy all six checks by construction. No LLM in the rank path
(network is off); this is deterministic Python that pulls only from parsed fields, so
hallucination is structurally impossible.

## The six checks (from submission_spec)

| Check | Requirement |
|---|---|
| Specific facts | Reference concrete fields: years, current title, named skills, signal values. |
| JD connection | Tie to a specific JD requirement, not generic praise. |
| Honest concerns | When there's a gap (notice period, location, missing skill), say so. |
| No hallucination | Every claim maps to a parsed field. |
| Variation | The 10 samples must read substantively differently — vary sentence frames AND which facts are selected. Not a name-insertion template. |
| Rank consistency | Tone matches rank: confident at rank 3, hedged/critical at rank 95. |

Penalized: empty, all-identical, name-insertion templates, hallucinated
skills/employers, reasoning that contradicts the rank. Spec guidance verbatim:
**"Don't try to be impressive; try to be specific and honest."**

## Implementation shape

- A library of ~10–15 sentence frames; pick frame + which 2–3 facts to surface using
  `random.Random(candidate_id)` so output is deterministic but varied across candidates.
- Always include: one concrete fact, one explicit JD connection, and — when present —
  one honest concern.
- Tie tone to rank band (e.g. top decile vs. filler near rank 100).