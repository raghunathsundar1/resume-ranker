---
title: Redrob Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# Redrob Ranker

Ranks 100,000 ML job candidates against a Senior ML Engineer (Search/Ranking/Recommendation) JD.

## Features
- **Top 100 Leaderboard** — filterable by score range and keyword search
- **Candidate Detail** — deep-dive on any top-100 candidate
- **Live Demo** — runs the full pipeline on the 50-candidate sample in real time
- **Architecture tab** — explains every design decision

## Reproduce locally
```bash
pip install -r requirements.txt
python normalize.py --input candidates.jsonl.gz --out features.parquet
python rank.py --candidates features.parquet --out submission.csv
python validate_submission.py submission.csv
```

Runtime: **33 seconds** for 100K candidates (CPU-only, no network).
