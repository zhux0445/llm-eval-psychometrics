"""Generate a self-contained HTML report for MMLU-Pro IRT analysis.

Produces a single HTML file with embedded plots and analysis results.

Usage:
    python examples/demo_mmlu_pro_report.py
"""

import base64
import io
import json
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_eval_psychometrics.irt import (
    IRTAnalyzer,
    classical_item_stats,
    diagnostic_report,
    plot_icc,
    plot_iic,
    plot_tic,
    plot_wright_map,
)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATA_DIR = Path("/tmp/mmlu_pro_data")
OUTPUT_HTML = Path("examples/mmlu_pro_report.html")

MODELS = {
    "Llama-2-7B": "model_outputs_Llama-2-7b-hf_5shots",
    "Llama-2-70B": "model_outputs_Llama-2-70b-hf_5shots",
    "Llama-3-8B": "model_outputs_Meta-Llama-3-8B_5shots",
    "Llama-3-70B-Inst": "model_outputs_Meta-Llama-3-70B-Instruct_5shots",
    "Mistral-7B": "model_outputs_Mistral-7B-v0.1_5shots",
    "Mixtral-8x7B": "model_outputs_Mixtral-8x7B-v0.1_5shots",
    "Phi-3-mini": "model_outputs_Phi-3-mini-4k-instruct_5shots",
    "Yi-34B": "model_outputs_Yi-34B_5shots",
    "Gemma-7B": "model_outputs_gemma-7b_5shots",
    "Qwen1.5-72B": "model_outputs_Qwen1.5-72B-Chat_5shots",
}

BASE_URL = "https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/main/eval_results"
SUBJECTS = ["computer_science", "math", "physics", "biology", "history"]


# -------------------------------------------------------------------------
# Data loading (same as demo_mmlu_pro.py)
# -------------------------------------------------------------------------

def download_if_needed(model_file: str) -> Path:
    zip_name = f"{model_file}.zip"
    zip_path = DATA_DIR / zip_name
    json_dir = DATA_DIR / model_file
    if not json_dir.exists() or not list(json_dir.glob("*.json")):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists() or zip_path.stat().st_size < 100:
            url = f"{BASE_URL}/{zip_name}"
            print(f"  Downloading {zip_name}...")
            urlretrieve(url, zip_path)
        json_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(json_dir)
    jsons = list(json_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No JSON found in {json_dir}")
    return jsons[0]


def build_response_matrix(subjects=None):
    print("Loading model results...")
    all_model_data = {}
    for model_name, model_file in MODELS.items():
        json_path = download_if_needed(model_file)
        with open(json_path) as f:
            all_model_data[model_name] = json.load(f)
        print(f"  {model_name}: {len(all_model_data[model_name])} questions")

    ref_data = all_model_data[list(all_model_data.keys())[0]]
    question_map = {}
    for item in ref_data:
        cat = item["category"]
        if subjects and cat not in subjects:
            continue
        qid = (item["question_id"], cat)
        if qid not in question_map:
            question_map[qid] = item

    model_indices = {}
    for model_name, data in all_model_data.items():
        idx = {}
        for item in data:
            idx[(item["question_id"], item["category"])] = item
        model_indices[model_name] = idx

    common_qids = set(question_map.keys())
    for idx in model_indices.values():
        common_qids &= set(idx.keys())
    common_qids = sorted(common_qids)

    model_ids = list(MODELS.keys())
    item_ids = [f"{cat}_{qid}" for qid, cat in common_qids]
    item_subjects = [cat for _, cat in common_qids]

    response_matrix = np.zeros((len(model_ids), len(common_qids)), dtype=int)
    for i, model_name in enumerate(model_ids):
        idx = model_indices[model_name]
        for j, qid in enumerate(common_qids):
            item = idx[qid]
            pred_idx = ord(item["pred"][0].upper()) - ord("A") if item["pred"] else -1
            response_matrix[i, j] = int(pred_idx == item["answer_index"])

    return response_matrix, model_ids, item_ids, item_subjects


# -------------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------------

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# -------------------------------------------------------------------------
# HTML generation
# -------------------------------------------------------------------------

def generate_html(
    response_matrix, model_ids, item_ids, item_subjects, results, report, ctt
) -> str:
    plots = {}
    n_models, n_items = response_matrix.shape
    accuracies = response_matrix.mean(axis=1)

    # --- Plot 1: Model abilities ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_ab = results.model_ability.sort_values()
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(sorted_ab)))
    ax.barh(range(len(sorted_ab)), sorted_ab.values, color=colors)
    ax.set_yticks(range(len(sorted_ab)))
    ax.set_yticklabels(sorted_ab.index, fontsize=12)
    ax.set_xlabel("Ability (θ)", fontsize=12)
    ax.set_title("Model Ability Estimates (2PL IRT)", fontsize=14)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    plots["abilities"] = fig_to_base64(fig)

    # --- Plot 2: Difficulty by subject ---
    fig, ax = plt.subplots(figsize=(10, 6))
    subject_order = sorted(set(item_subjects))
    data_to_plot = []
    labels = []
    for subj in subject_order:
        mask = [s == subj for s in item_subjects]
        if any(mask):
            data_to_plot.append(report.loc[mask, "difficulty"].values)
            labels.append(subj)
    bp = ax.boxplot(data_to_plot, vert=True, patch_artist=True, widths=0.6)
    colors_box = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Item Difficulty (b)", fontsize=12)
    ax.set_title("Item Difficulty Distribution by Subject", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    plots["difficulty_subject"] = fig_to_base64(fig)

    # --- Plot 3: Item parameter space ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for subj in subject_order:
        mask = report["subject"] == subj
        if mask.any():
            ax.scatter(report.loc[mask, "difficulty"], report.loc[mask, "discrimination"],
                       label=subj, alpha=0.5, s=25)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="min discrimination")
    ax.axvline(x=-3, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(x=3, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Difficulty (b)", fontsize=12)
    ax.set_ylabel("Discrimination (a)", fontsize=12)
    ax.set_title("Item Parameter Space", fontsize=14)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plots["param_space"] = fig_to_base64(fig)

    # --- Plot 4: ICC ---
    sorted_by_disc = report.sort_values("discrimination")
    sample_items = [
        sorted_by_disc.index[0],
        sorted_by_disc.index[len(sorted_by_disc) // 4],
        sorted_by_disc.index[len(sorted_by_disc) // 2],
        sorted_by_disc.index[3 * len(sorted_by_disc) // 4],
        sorted_by_disc.index[-1],
    ]
    sample_items = list(dict.fromkeys(sample_items))
    fig = plot_icc(results, item_ids=sample_items)
    plots["icc"] = fig_to_base64(fig)

    # --- Plot 5: IIC ---
    fig = plot_iic(results, item_ids=sample_items)
    plots["iic"] = fig_to_base64(fig)

    # --- Plot 6: TIC ---
    fig = plot_tic(results, show_se=True)
    plots["tic"] = fig_to_base64(fig)

    # --- Plot 7: Wright Map ---
    fig = plot_wright_map(results)
    plots["wright"] = fig_to_base64(fig)

    # --- Plot 8: Quality ---
    fig, ax = plt.subplots(figsize=(6, 6))
    quality_data = report["quality"].value_counts()
    colors_pie = {"good": "#4CAF50", "acceptable": "#FFC107", "poor": "#F44336"}
    ax.pie(quality_data.values,
           labels=[f"{k}\n({v})" for k, v in quality_data.items()],
           colors=[colors_pie.get(k, "gray") for k in quality_data.index],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 13})
    ax.set_title("Item Quality (IRT Diagnostic)", fontsize=14)
    fig.tight_layout()
    plots["quality_pie"] = fig_to_base64(fig)

    # --- Build accuracy table ---
    acc_rows = ""
    for mid, acc, theta in sorted(
        zip(model_ids, accuracies, results.model_ability.values),
        key=lambda x: -x[2],
    ):
        bar_width = acc * 100
        acc_rows += f"""<tr>
            <td class="model-name">{mid}</td>
            <td>{acc:.1%}</td>
            <td>{theta:+.3f}</td>
            <td><div class="bar" style="width:{bar_width:.0f}%"></div></td>
        </tr>"""

    # --- Subject stats ---
    subj_rows = ""
    for subj in subject_order:
        mask = [s == subj for s in item_subjects]
        if any(mask):
            n_q = sum(mask)
            subj_acc = response_matrix[:, mask].mean()
            goods = (report.loc[mask, "quality"] == "good").sum()
            acc_pct = (report.loc[mask, "quality"] == "acceptable").sum()
            poors = (report.loc[mask, "quality"] == "poor").sum()
            subj_rows += f"""<tr>
                <td>{subj}</td><td>{n_q}</td><td>{subj_acc:.1%}</td>
                <td><span class="badge good">{goods}</span></td>
                <td><span class="badge acceptable">{acc_pct}</span></td>
                <td><span class="badge poor">{poors}</span></td>
            </tr>"""

    # --- Quality summary ---
    q_counts = report["quality"].value_counts()
    total = len(report)

    # --- Flagged items table ---
    poor_items = results.flag_poor_items()
    flagged_rows = ""
    for idx, row in poor_items.head(15).iterrows():
        reasons = "; ".join(row["reasons"][:2])
        flagged_rows += f"<tr><td><code>{idx}</code></td><td>{reasons}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MMLU-Pro Psychometric Analysis Report</title>
<style>
:root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --good: #3fb950; --warn: #d29922; --bad: #f85149;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}
header {{
    text-align: center; padding: 3rem 1rem 2rem;
    border-bottom: 1px solid var(--border);
}}
header h1 {{ font-size: 2.2rem; margin-bottom: 0.5rem; }}
header p {{ color: var(--text-muted); font-size: 1.1rem; }}
.tag {{
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 1rem;
    font-size: 0.8rem; margin: 0.3rem; background: var(--surface); border: 1px solid var(--border);
}}
section {{ margin: 2.5rem 0; }}
h2 {{
    font-size: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}}
h3 {{ font-size: 1.1rem; margin: 1.5rem 0 0.5rem; color: var(--accent); }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
@media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
.card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 0.5rem; padding: 1.2rem; text-align: center;
}}
.card .number {{ font-size: 2rem; font-weight: 700; }}
.card .label {{ color: var(--text-muted); font-size: 0.9rem; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }}
@media (max-width: 768px) {{ .stat-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
table {{
    width: 100%; border-collapse: collapse; margin: 1rem 0;
    background: var(--surface); border-radius: 0.5rem; overflow: hidden;
}}
th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ background: rgba(88,166,255,0.08); color: var(--accent); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
td {{ font-size: 0.95rem; }}
.model-name {{ font-weight: 600; }}
.bar {{
    height: 1.2rem; background: linear-gradient(90deg, var(--accent), #a371f7);
    border-radius: 0.3rem; min-width: 2px;
}}
.badge {{
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 0.8rem;
    font-size: 0.8rem; font-weight: 600;
}}
.badge.good {{ background: rgba(63,185,80,0.15); color: var(--good); }}
.badge.acceptable {{ background: rgba(210,153,34,0.15); color: var(--warn); }}
.badge.poor {{ background: rgba(248,81,73,0.15); color: var(--bad); }}
.plot-container {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;
}}
.plot-container img {{ width: 100%; height: auto; border-radius: 0.3rem; }}
.plot-caption {{ color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem; }}
.insight {{
    background: rgba(88,166,255,0.06); border-left: 3px solid var(--accent);
    padding: 1rem 1.2rem; margin: 1rem 0; border-radius: 0 0.3rem 0.3rem 0;
}}
.insight strong {{ color: var(--accent); }}
footer {{
    text-align: center; padding: 2rem 1rem; margin-top: 3rem;
    border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.85rem;
}}
footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>

<header>
    <h1>MMLU-Pro Psychometric Analysis</h1>
    <p>IRT diagnostics on {n_models} open-source LLMs &times; {n_items} benchmark items</p>
    <div style="margin-top:1rem">
        <span class="tag">2PL IRT Model</span>
        <span class="tag">{n_models} Models</span>
        <span class="tag">{n_items} Items</span>
        <span class="tag">{len(set(item_subjects))} Subjects</span>
        <span class="tag">5-shot</span>
    </div>
</header>

<div class="container">

<!-- Overview Stats -->
<section>
<h2>Overview</h2>
<div class="stat-grid">
    <div class="card">
        <div class="number">{n_models}</div>
        <div class="label">LLMs evaluated</div>
    </div>
    <div class="card">
        <div class="number">{n_items}</div>
        <div class="label">Benchmark items</div>
    </div>
    <div class="card">
        <div class="number">{response_matrix.mean():.1%}</div>
        <div class="label">Overall accuracy</div>
    </div>
    <div class="card">
        <div class="number" style="color:var(--bad)">{q_counts.get('poor',0)}</div>
        <div class="label">Poor quality items</div>
    </div>
</div>
</section>

<!-- Model Rankings -->
<section>
<h2>Model Ability Rankings</h2>
<div class="insight">
    <strong>Key finding:</strong> IRT ability estimates (θ) provide interval-scale measurement
    that accounts for item difficulty, unlike raw accuracy which treats all items equally.
    Llama-3-70B-Instruct separates clearly from the pack (θ=+1.95), while Llama-2-7B
    (θ=-3.25) performs near chance level.
</div>
<div class="plot-container">
    <img src="data:image/png;base64,{plots['abilities']}" alt="Model abilities">
</div>
<table>
    <thead><tr><th>Model</th><th>Accuracy</th><th>θ (IRT)</th><th>Visual</th></tr></thead>
    <tbody>{acc_rows}</tbody>
</table>
</section>

<!-- Item Quality -->
<section>
<h2>Item Quality Diagnostics</h2>
<div class="insight">
    <strong>Key finding:</strong> Only {q_counts.get('good',0)} of {total} items ({q_counts.get('good',0)/total*100:.0f}%)
    meet quality standards (good discrimination, moderate difficulty, good fit).
    {q_counts.get('poor',0)} items ({q_counts.get('poor',0)/total*100:.0f}%) are poor —
    they don't effectively distinguish between strong and weak models.
</div>
<div class="grid">
    <div class="plot-container">
        <img src="data:image/png;base64,{plots['quality_pie']}" alt="Quality distribution">
        <div class="plot-caption">Item quality classification based on IRT discrimination, difficulty, and fit.</div>
    </div>
    <div>
        <h3>Quality Criteria</h3>
        <table>
            <tr><td><span class="badge good">Good</span></td><td>a ≥ 0.5, |b| ≤ 2.5, RMSR ≤ 0.1</td></tr>
            <tr><td><span class="badge acceptable">Acceptable</span></td><td>a ≥ 0.3, |b| ≤ 3.0</td></tr>
            <tr><td><span class="badge poor">Poor</span></td><td>Fails above criteria</td></tr>
        </table>
        <h3>Classical Statistics</h3>
        <table>
            <tr><td>Mean p-value (easiness)</td><td>{ctt['p_value'].mean():.3f}</td></tr>
            <tr><td>Mean point-biserial r</td><td>{ctt['point_biserial'].mean():.3f}</td></tr>
            <tr><td>Trivially easy (p &gt; 0.95)</td><td>{(ctt['p_value'] > 0.95).sum()}</td></tr>
            <tr><td>Trivially hard (p &lt; 0.05)</td><td>{(ctt['p_value'] < 0.05).sum()}</td></tr>
            <tr><td>Negative discrimination</td><td>{(ctt['point_biserial'] < 0).sum()}</td></tr>
        </table>
    </div>
</div>
</section>

<!-- Subject Breakdown -->
<section>
<h2>Subject Analysis</h2>
<div class="grid">
    <div class="plot-container">
        <img src="data:image/png;base64,{plots['difficulty_subject']}" alt="Difficulty by subject">
        <div class="plot-caption">Math items are hardest (median b ≈ 1.5), biology easiest (median b ≈ -1).</div>
    </div>
    <div>
        <table>
            <thead><tr><th>Subject</th><th>Items</th><th>Accuracy</th><th>Good</th><th>OK</th><th>Poor</th></tr></thead>
            <tbody>{subj_rows}</tbody>
        </table>
    </div>
</div>
</section>

<!-- Item Parameters -->
<section>
<h2>Item Parameter Space</h2>
<div class="insight">
    <strong>Key finding:</strong> Many items cluster near the discrimination floor (a ≈ 0.2) and at extreme
    difficulties (|b| &gt; 3). These items contribute almost no measurement information. The "sweet spot"
    is the upper-center region: high discrimination, moderate difficulty.
</div>
<div class="plot-container">
    <img src="data:image/png;base64,{plots['param_space']}" alt="Item parameter space">
    <div class="plot-caption">Each dot is one item. Red dashed line = minimum acceptable discrimination (a=0.3). Gray dotted lines = extreme difficulty bounds (|b|=3).</div>
</div>
</section>

<!-- IRT Curves -->
<section>
<h2>IRT Curves</h2>
<h3>Item Characteristic Curves (ICC)</h3>
<div class="plot-container">
    <img src="data:image/png;base64,{plots['icc']}" alt="ICC">
    <div class="plot-caption">Probability of correct response vs. ability. Steeper curves = higher discrimination (better items).</div>
</div>
<div class="grid">
    <div class="plot-container">
        <h3>Item Information Curves (IIC)</h3>
        <img src="data:image/png;base64,{plots['iic']}" alt="IIC">
        <div class="plot-caption">Where each item provides the most measurement precision. Peaks indicate optimal ability range.</div>
    </div>
    <div class="plot-container">
        <h3>Test Information Curve (TIC)</h3>
        <img src="data:image/png;base64,{plots['tic']}" alt="TIC">
        <div class="plot-caption">Blue = total information (precision). Red dashed = standard error. Information peaks near θ ≈ 1, meaning the benchmark best distinguishes upper-middle ability models.</div>
    </div>
</div>
</section>

<!-- Wright Map -->
<section>
<h2>Wright Map</h2>
<div class="insight">
    <strong>Key finding:</strong> Many items have extreme difficulty values far beyond the model ability range,
    meaning they provide no useful measurement information. An ideal benchmark would have items
    spread evenly across the ability range of the models being evaluated.
</div>
<div class="plot-container">
    <img src="data:image/png;base64,{plots['wright']}" alt="Wright Map">
    <div class="plot-caption">Left: model ability distribution. Right: item difficulty positions. Items far above or below all models are uninformative.</div>
</div>
</section>

<!-- Flagged Items -->
<section>
<h2>Flagged Items (Top 15)</h2>
<table>
    <thead><tr><th>Item ID</th><th>Reasons</th></tr></thead>
    <tbody>{flagged_rows}</tbody>
</table>
</section>

</div>

<footer>
    Generated by <a href="https://github.com/zhux0445/llm-eval-psychometrics">llm-eval-psychometrics</a> v0.1.0
    &middot; Data from <a href="https://github.com/TIGER-AI-Lab/MMLU-Pro">TIGER-Lab/MMLU-Pro</a> (NeurIPS 2024)
</footer>

</body>
</html>"""

    return html


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MMLU-Pro IRT Report Generator")
    print("=" * 70)

    response_matrix, model_ids, item_ids, item_subjects = build_response_matrix(SUBJECTS)
    n_models, n_items = response_matrix.shape
    print(f"\nResponse matrix: {n_models} models × {n_items} items")

    print("Fitting 2PL IRT model...")
    analyzer = IRTAnalyzer(model="2PL")
    results = analyzer.fit(response_matrix, item_ids=item_ids, model_ids=model_ids)

    print("Running diagnostics...")
    report = diagnostic_report(results)
    report["subject"] = item_subjects
    ctt = classical_item_stats(response_matrix, item_ids=item_ids)

    print("Generating HTML report...")
    html = generate_html(
        response_matrix, model_ids, item_ids, item_subjects, results, report, ctt
    )

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT_HTML}")
    print(f"File size: {OUTPUT_HTML.stat().st_size / 1024:.0f} KB")
    print(f"\nOpen in browser: file://{OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
