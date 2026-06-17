"""Hugging Face Spaces demo for Redrob Ranker.

Displays the pre-computed top-100 submission and lets users browse
candidates with reasoning. No live ranking (the 100K run is offline).
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

APP_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load pre-computed results
# ---------------------------------------------------------------------------

SUBMISSION_PATH = APP_DIR / "submission.csv"
SAMPLE_PATH = APP_DIR / "data" / "sample_candidates.json"


def _load_submission() -> pd.DataFrame:
    if not SUBMISSION_PATH.exists():
        return pd.DataFrame(columns=["rank", "candidate_id", "score", "reasoning"])
    with open(SUBMISSION_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float).round(4)
    return df.sort_values("rank").reset_index(drop=True)


def _load_sample_profiles() -> dict:
    if not SAMPLE_PATH.exists():
        return {}
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {c["candidate_id"]: c for c in data}


DF = _load_submission()
PROFILES = _load_sample_profiles()

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def get_leaderboard(score_min: float, score_max: float, search: str) -> pd.DataFrame:
    df = DF.copy()
    df = df[(df["score"] >= score_min) & (df["score"] <= score_max)]
    if search.strip():
        mask = (df["candidate_id"].str.contains(search, case=False) |
                df["reasoning"].str.contains(search, case=False))
        df = df[mask]
    return df[["rank", "candidate_id", "score", "reasoning"]].head(100)


def get_candidate_detail(candidate_id: str) -> str:
    row = DF[DF["candidate_id"] == candidate_id]
    if row.empty:
        return "Candidate not found in top-100."

    r = row.iloc[0]
    lines = [
        f"## Rank {int(r['rank'])} — {candidate_id}",
        f"**Score:** {float(r['score']):.4f}",
        "",
        f"**Reasoning:** {r['reasoning']}",
    ]

    if candidate_id in PROFILES:
        p = PROFILES[candidate_id]
        prof = p.get("profile", {})
        lines += [
            "",
            "---",
            f"**Name:** {prof.get('anonymized_name', 'N/A')}",
            f"**Title:** {prof.get('current_title', 'N/A')} @ {prof.get('current_company', 'N/A')}",
            f"**Experience:** {prof.get('years_of_experience', 'N/A')} years",
            f"**Location:** {prof.get('location', 'N/A')}, {prof.get('country', 'N/A')}",
            f"**Industry:** {prof.get('current_industry', 'N/A')}",
            "",
            f"**Summary:** {(prof.get('summary') or '')[:400]}...",
        ]
        skills = p.get("skills", [])[:8]
        if skills:
            skill_str = ", ".join(
                f"{s['name']} ({s.get('proficiency','?')})" for s in skills
            )
            lines.append(f"\n**Top Skills:** {skill_str}")
    else:
        lines.append("\n*(Full profile only available for sample candidates)*")

    return "\n".join(lines)


def _run_pipeline(input_path: str, json_array: bool,
                  parquet_out: str, csv_out: str,
                  timeout_norm: int = 300, timeout_rank: int = 600
                  ) -> tuple[pd.DataFrame, str]:
    """Shared helper: normalize → rank → DataFrame."""
    cmd_norm = [sys.executable, "normalize.py",
                "--input", input_path, "--out", parquet_out]
    if json_array:
        cmd_norm.append("--json-array")
    r1 = subprocess.run(cmd_norm, capture_output=True, text=True,
                        timeout=timeout_norm, cwd=APP_DIR)
    if r1.returncode != 0:
        return pd.DataFrame(), f"normalize.py failed:\n{r1.stderr}"

    r2 = subprocess.run(
        [sys.executable, "rank.py", "--candidates", parquet_out, "--out", csv_out],
        capture_output=True, text=True, timeout=timeout_rank, cwd=APP_DIR,
    )
    if r2.returncode != 0:
        return pd.DataFrame(), f"rank.py failed:\n{r2.stderr}"

    csv_full = Path(APP_DIR) / csv_out
    with open(csv_full, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(rows)
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float).round(4)
    log = (r1.stderr + "\n" + r2.stderr).strip()
    return df.sort_values("rank").reset_index(drop=True), log


def run_sample_rank() -> tuple[pd.DataFrame, str]:
    try:
        return _run_pipeline(
            str(APP_DIR / "data" / "sample_candidates.json"), json_array=True,
            parquet_out="sample_demo.parquet", csv_out="sample_demo_submission.csv",
            timeout_norm=60, timeout_rank=120,
        )
    except Exception as e:
        return pd.DataFrame(), str(e)


def run_uploaded(file_obj) -> tuple[pd.DataFrame, str]:
    """Accept a user-uploaded JSONL / JSON file and run the full pipeline."""
    if file_obj is None:
        return pd.DataFrame(), "No file uploaded."
    try:
        # Gradio 5 returns FileData(path=<tmp>, orig_name=<original filename>)
        # Gradio 4 returned an object where .name was the tmp path (same ext as original)
        if isinstance(file_obj, str):
            path, orig_name = file_obj, Path(file_obj).name
        elif hasattr(file_obj, "path"):          # Gradio 5
            path = file_obj.path
            orig_name = getattr(file_obj, "orig_name", None) or Path(path).name
        else:                                    # Gradio 4 fallback
            path = file_obj.name
            orig_name = Path(path).name

        orig_lower = (orig_name or "").lower()
        if orig_lower.endswith(".json"):
            json_array = True
        elif orig_lower.endswith(".jsonl") or orig_lower.endswith(".jsonl.gz") or orig_lower.endswith(".gz"):
            json_array = False
        else:
            return pd.DataFrame(), (
                f"Unsupported extension '{Path(orig_name).suffix}'. "
                "Upload .json (JSON array), .jsonl, or .jsonl.gz"
            )
        return _run_pipeline(
            path, json_array=json_array,
            parquet_out="upload_demo.parquet", csv_out="upload_submission.csv",
        )
    except Exception as e:
        return pd.DataFrame(), str(e)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Redrob Ranker — Top 100 ML Engineers") as demo:
    gr.Markdown("""
# Redrob Ranker
**Ranks 100,000 ML candidates against a Senior ML Engineer (Search/Ranking/Recommendation) JD.**

- **B2 Integrity gate:** removes honeypots (impossible tenure, expert-0-months, future dates) and JD mismatches (services-only, non-ML titles)
- **B3 Hybrid relevance:** 40% BM25 + 60% evidence-weighted concept scoring (uncorroborated skill claims penalised 10×)
- **B4 Composite:** `relevance^0.4 · fit^0.15 · behavioral^0.1 · logistics^0.05 · (1−penalty)`
- **B5 Reasoning:** deterministic, per-candidate, seeded by candidate_id — no LLM

Runtime: **33 seconds** on 100K candidates (CPU-only, no network).
    """)

    with gr.Tabs():
        # --- Tab 1: Full leaderboard ---
        with gr.Tab("Top 100 Leaderboard"):
            with gr.Row():
                score_min = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Min score")
                score_max = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Max score")
                search_box = gr.Textbox(label="Search (candidate ID or reasoning keyword)", scale=2)
            filter_btn = gr.Button("Apply filter")
            leaderboard = gr.DataFrame(
                value=DF[["rank", "candidate_id", "score", "reasoning"]].head(100),
                label="Top 100",
                interactive=False,
                wrap=True,
            )
            filter_btn.click(
                get_leaderboard,
                inputs=[score_min, score_max, search_box],
                outputs=leaderboard,
            )

        # --- Tab 2: Candidate detail ---
        with gr.Tab("Candidate Detail"):
            cid_input = gr.Textbox(
                label="Candidate ID (e.g. CAND_0030953)",
                placeholder="CAND_0030953",
            )
            detail_btn = gr.Button("Look up")
            detail_out = gr.Markdown()
            detail_btn.click(get_candidate_detail, inputs=cid_input, outputs=detail_out)
            gr.Markdown("*Full profiles available only for the 50-candidate sample set.*")

        # --- Tab 3: Upload your own file ---
        with gr.Tab("Upload & Rank"):
            gr.Markdown("""
### Upload your own candidates file
Accepts **JSON array** (`.json`), **JSONL** (`.jsonl`), or **gzipped JSONL** (`.jsonl.gz`).
Each record must have `candidate_id` and the standard Redrob profile fields.
The full pipeline runs on your file: normalize → integrity gate → score → rank.
Results are returned as the top-ranked candidates with per-candidate reasoning.
            """)
            with gr.Row():
                upload_file = gr.File(
                    label="Candidates file (.json / .jsonl / .jsonl.gz)",
                    file_types=[".json", ".jsonl", ".gz"],
                )
                with gr.Column():
                    upload_btn = gr.Button("Run pipeline", variant="primary")
                    upload_download = gr.File(label="Download results CSV", visible=False)
            upload_log = gr.Textbox(label="Pipeline log", lines=6, interactive=False)
            upload_table = gr.DataFrame(label="Ranked results", interactive=False, wrap=True)

            def run_and_expose(file_obj):
                df, log = run_uploaded(file_obj)
                csv_path = APP_DIR / "upload_submission.csv"
                visible = csv_path.exists() and not df.empty
                return df, log, gr.File(value=str(csv_path) if visible else None,
                                        visible=visible)

            upload_btn.click(
                run_and_expose,
                inputs=upload_file,
                outputs=[upload_table, upload_log, upload_download],
            )

        # --- Tab 4: Live demo on built-in sample ---
        with gr.Tab("Live Demo (50-candidate sample)"):
            gr.Markdown("""
Run the full pipeline on the built-in 50-candidate sample in real time.
Demonstrates the end-to-end flow: normalize → gate → score → rank → reason.
            """)
            run_btn = gr.Button("Run pipeline on sample", variant="primary")
            demo_log = gr.Textbox(label="Pipeline log", lines=8, interactive=False)
            demo_table = gr.DataFrame(label="Sample ranking results", interactive=False, wrap=True)
            run_btn.click(run_sample_rank, outputs=[demo_table, demo_log])

        # Architecture tab removed — details available during live interview

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
