# Judging, submission format, and repo rules

## The 5 stages

1. **Auto-validation** — format checks (see rejections below).
2. **Metric scoring** — top-100 ranking vs. hidden ground truth (NDCG etc.).
3. **Code reproduction** — judges run your single documented command under the
   compute limits; must reproduce the CSV. Also the honeypot-rate DQ filter (>10%).
4. **Manual review** — reasoning quality (see `reasoning.md`), methodology coherence,
   **git history authenticity**, code quality.
5. **Defend-your-work interview** — top-X finalists, 30-min call; explain architecture
   and defend choices. Cannot fake what you didn't build.

3 submissions max, last valid one counts. No live leaderboard. Sandbox link required.
AI tools allowed if declared; judging is built so real-engineering-with-AI passes and
AI-only dumps fail at 3–5.

## Submission CSV format

Header + exactly 100 data rows. Columns: `candidate_id,rank,score,reasoning`.

- rank 1..100 each once; candidate_id unique and present in the pool.
- score **non-increasing** with rank (ties allowed). Unique ranks even on ties →
  break by a secondary signal or by **candidate_id ascending**.
- `reasoning` optional but heavily recommended (drives Stage 4).
- `rank.py` self-validates before writing: re-implement these checks AND shell out to
  `validate_submission.py`; refuse to emit an invalid file.

### Auto-validator rejections — guard all of these
99/101 rows; ranks starting at 0; duplicate ids; id not in pool; all scores identical;
score increasing with rank; file as .xlsx/.json instead of .csv.

## Git history (graded at Stage 4)

The spec grades **"git history authenticity (real iteration vs single dump)"** and
names **"flat git history with no iteration"** as a failure mode. It does NOT define a
separate rubric for commit-*message wording* — clear messages are how a human reviewer
is convinced the work evolved, not the metric itself. So:

- Commit per logical step from day one (normalize, ontology v1, gate v1, tune, fix).
- Messages narrate what changed and why, citing local eval deltas where relevant.
  Good: `tune skill evidence mult 0.1->0.15; local NDCG@10 0.71->0.74`
  Bad: `update`, `wip`, `final`.
- History must match the code and survive interview cross-examination.
- Never squash the meaningful history into one commit. The iteration is the evidence.

## Repo must include (Stage 3)

README with the single reproduce command; full source (no hidden/manual steps);
precomputed artifacts or a script that builds them; pinned `requirements.txt`;
`submission_metadata.yaml` at root (from the bundle template).