"""End-to-end IRT analysis on real MMLU-Pro benchmark data.

Uses per-question evaluation results from 10 open-source LLMs on MMLU-Pro
(TIGER-Lab, NeurIPS 2024) to demonstrate psychometric diagnostics.

Data source: https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main/eval_results

Usage:
    python examples/demo_mmlu_pro.py
"""

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
OUTPUT_DIR = Path("examples/mmlu_pro_plots")

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

# Focus on a few subjects for cleaner IRT analysis
SUBJECTS = ["computer_science", "math", "physics", "biology", "history"]


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def download_if_needed(model_file: str) -> Path:
    """Download and extract model results if not already cached."""
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

    # Find the json file inside
    jsons = list(json_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No JSON found in {json_dir}")
    return jsons[0]


def load_model_results(json_path: Path) -> list[dict]:
    """Load per-question results from a model's JSON file."""
    with open(json_path) as f:
        return json.load(f)


def build_response_matrix(
    subjects: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build a binary response matrix from MMLU-Pro model results.

    Returns:
        (response_matrix, model_ids, item_ids, item_subjects)
    """
    print("Loading model results...")

    # Load all models' data
    all_model_data = {}
    for model_name, model_file in MODELS.items():
        json_path = download_if_needed(model_file)
        all_model_data[model_name] = load_model_results(json_path)
        print(f"  {model_name}: {len(all_model_data[model_name])} questions")

    # Use the first model to define the question set (by question_id + category)
    ref_model = list(all_model_data.keys())[0]
    ref_data = all_model_data[ref_model]

    # Build question index: (question_id, category) -> index in ref_data
    # Filter by subjects if specified
    question_map = {}
    for item in ref_data:
        cat = item["category"]
        if subjects and cat not in subjects:
            continue
        qid = (item["question_id"], cat)
        if qid not in question_map:
            question_map[qid] = item

    print(f"\n  Selected {len(question_map)} questions across {len(set(q[1] for q in question_map))} subjects")

    # For each model, build index for fast lookup
    model_indices = {}
    for model_name, data in all_model_data.items():
        idx = {}
        for item in data:
            qid = (item["question_id"], item["category"])
            idx[qid] = item
        model_indices[model_name] = idx

    # Find common questions across ALL models
    common_qids = set(question_map.keys())
    for model_name, idx in model_indices.items():
        common_qids &= set(idx.keys())

    common_qids = sorted(common_qids)
    print(f"  Common questions across all models: {len(common_qids)}")

    # Build response matrix
    model_ids = list(MODELS.keys())
    item_ids = [f"{cat}_{qid}" for qid, cat in common_qids]
    item_subjects = [cat for _, cat in common_qids]

    response_matrix = np.zeros((len(model_ids), len(common_qids)), dtype=int)

    for i, model_name in enumerate(model_ids):
        idx = model_indices[model_name]
        for j, qid in enumerate(common_qids):
            item = idx[qid]
            pred = item["pred"]
            answer_idx = item["answer_index"]
            # pred is a letter (A-J), answer_index is 0-based int
            pred_idx = ord(pred[0].upper()) - ord("A") if pred else -1
            response_matrix[i, j] = int(pred_idx == answer_idx)

    return response_matrix, model_ids, item_ids, item_subjects


# -------------------------------------------------------------------------
# Main demo
# -------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 1. Load real MMLU-Pro data
    # =================================================================
    print("=" * 70)
    print("MMLU-Pro IRT Analysis: 10 Open-Source LLMs")
    print("=" * 70)

    response_matrix, model_ids, item_ids, item_subjects = build_response_matrix(
        subjects=SUBJECTS
    )
    n_models, n_items = response_matrix.shape

    print(f"\nResponse matrix: {n_models} models × {n_items} items")

    # Per-model accuracy
    print("\nModel accuracies:")
    accuracies = response_matrix.mean(axis=1)
    for mid, acc in sorted(zip(model_ids, accuracies), key=lambda x: -x[1]):
        bar = "█" * int(acc * 40)
        print(f"  {mid:<20s} {acc:.1%} {bar}")

    # Per-subject accuracy
    print("\nSubject accuracies:")
    for subj in SUBJECTS:
        mask = [s == subj for s in item_subjects]
        if any(mask):
            subj_acc = response_matrix[:, mask].mean()
            n_q = sum(mask)
            print(f"  {subj:<20s} {subj_acc:.1%} ({n_q} items)")

    # =================================================================
    # 2. Fit 2PL IRT model
    # =================================================================
    print("\n" + "=" * 70)
    print("Fitting 2PL IRT Model")
    print("=" * 70)

    analyzer = IRTAnalyzer(model="2PL")
    results = analyzer.fit(response_matrix, item_ids=item_ids, model_ids=model_ids)

    print("\nModel ability estimates (θ):")
    sorted_abilities = results.model_ability.sort_values(ascending=False)
    for mid, theta in sorted_abilities.items():
        print(f"  {mid:<20s} θ = {theta:+.3f}")

    print(f"\nItem parameter summary:")
    print(f"  Discrimination: mean={results.item_params['discrimination'].mean():.3f}, "
          f"range=[{results.item_params['discrimination'].min():.3f}, "
          f"{results.item_params['discrimination'].max():.3f}]")
    print(f"  Difficulty:      mean={results.item_params['difficulty'].mean():.3f}, "
          f"range=[{results.item_params['difficulty'].min():.3f}, "
          f"{results.item_params['difficulty'].max():.3f}]")

    # =================================================================
    # 3. Item diagnostics
    # =================================================================
    print("\n" + "=" * 70)
    print("Item Diagnostics")
    print("=" * 70)

    report = diagnostic_report(results)

    quality_counts = report["quality"].value_counts()
    print(f"\nItem quality distribution:")
    total = len(report)
    for q in ["good", "acceptable", "poor"]:
        count = quality_counts.get(q, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {q:<12s} {count:>4d} ({pct:4.1f}%) {bar}")

    # Per-subject quality breakdown
    report["subject"] = item_subjects
    print("\nQuality by subject:")
    for subj in SUBJECTS:
        subj_report = report[report["subject"] == subj]
        if len(subj_report) == 0:
            continue
        goods = (subj_report["quality"] == "good").sum()
        acceptable = (subj_report["quality"] == "acceptable").sum()
        poors = (subj_report["quality"] == "poor").sum()
        print(f"  {subj:<20s} good={goods}, acceptable={acceptable}, poor={poors}")

    # Flag worst items
    poor = results.flag_poor_items(min_discrimination=0.3, max_difficulty_abs=3.0)
    print(f"\n{len(poor)} items flagged as problematic ({len(poor)/total*100:.0f}%)")
    if len(poor) > 0:
        print("Top 10 worst items:")
        for idx, row in poor.head(10).iterrows():
            print(f"  {idx}: {', '.join(row['reasons'][:2])}")

    # =================================================================
    # 4. Classical test theory stats
    # =================================================================
    print("\n" + "=" * 70)
    print("Classical Test Theory Statistics")
    print("=" * 70)

    ctt = classical_item_stats(response_matrix, item_ids=item_ids)
    print(f"\n  Mean p-value (item easiness): {ctt['p_value'].mean():.3f}")
    print(f"  Mean point-biserial r:        {ctt['point_biserial'].mean():.3f}")

    # Items everyone gets right or wrong (uninformative)
    trivial_easy = (ctt["p_value"] > 0.95).sum()
    trivial_hard = (ctt["p_value"] < 0.05).sum()
    negative_rpb = (ctt["point_biserial"] < 0).sum()
    print(f"\n  Trivially easy (p > 0.95): {trivial_easy}")
    print(f"  Trivially hard (p < 0.05): {trivial_hard}")
    print(f"  Negative discrimination:   {negative_rpb}")

    # =================================================================
    # 5. Generate plots
    # =================================================================
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    # --- Plot 1: Model ability ranking ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_ab = results.model_ability.sort_values()
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(sorted_ab)))
    ax.barh(range(len(sorted_ab)), sorted_ab.values, color=colors)
    ax.set_yticks(range(len(sorted_ab)))
    ax.set_yticklabels(sorted_ab.index)
    ax.set_xlabel("Ability (θ)")
    ax.set_title("MMLU-Pro: Model Ability Estimates (2PL IRT)")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_abilities.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/model_abilities.png")
    plt.close(fig)

    # --- Plot 2: Item difficulty distribution by subject ---
    fig, ax = plt.subplots(figsize=(10, 6))
    report_with_params = report.copy()
    subject_order = sorted(SUBJECTS)
    positions = []
    labels = []
    data_to_plot = []
    for i, subj in enumerate(subject_order):
        mask = report_with_params["subject"] == subj
        if mask.any():
            data_to_plot.append(report_with_params.loc[mask, "difficulty"].values)
            positions.append(i)
            labels.append(subj)

    bp = ax.boxplot(data_to_plot, positions=positions, vert=True, patch_artist=True, widths=0.6)
    colors_box = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Item Difficulty (b)")
    ax.set_title("MMLU-Pro: Item Difficulty Distribution by Subject")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "difficulty_by_subject.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/difficulty_by_subject.png")
    plt.close(fig)

    # --- Plot 3: Discrimination vs Difficulty scatter ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for subj in SUBJECTS:
        mask = report["subject"] == subj
        if mask.any():
            ax.scatter(
                report.loc[mask, "difficulty"],
                report.loc[mask, "discrimination"],
                label=subj, alpha=0.5, s=20,
            )
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="min discrimination")
    ax.axvline(x=-3, color="gray", linestyle=":", alpha=0.4)
    ax.axvline(x=3, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Difficulty (b)")
    ax.set_ylabel("Discrimination (a)")
    ax.set_title("MMLU-Pro: Item Parameter Space")
    ax.legend(fontsize="small", loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "item_parameter_space.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/item_parameter_space.png")
    plt.close(fig)

    # --- Plot 4: ICC for a few interesting items ---
    # Pick items with varying discrimination: high, medium, low
    sorted_by_disc = report.sort_values("discrimination")
    low_disc_item = sorted_by_disc.index[0]
    mid_disc_item = sorted_by_disc.index[len(sorted_by_disc) // 2]
    high_disc_item = sorted_by_disc.index[-1]
    # Also pick one good & one poor item
    good_items = report[report["quality"] == "good"]
    poor_items = report[report["quality"] == "poor"]
    sample_items = [low_disc_item, mid_disc_item, high_disc_item]
    if len(good_items) > 0:
        sample_items.append(good_items.index[0])
    if len(poor_items) > 0:
        sample_items.append(poor_items.index[0])
    sample_items = list(dict.fromkeys(sample_items))[:5]  # deduplicate

    fig = plot_icc(results, item_ids=sample_items)
    fig.savefig(OUTPUT_DIR / "icc_selected.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/icc_selected.png")
    plt.close(fig)

    # --- Plot 5: IIC for same items ---
    fig = plot_iic(results, item_ids=sample_items)
    fig.savefig(OUTPUT_DIR / "iic_selected.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/iic_selected.png")
    plt.close(fig)

    # --- Plot 6: Test Information Curve ---
    fig = plot_tic(results, show_se=True)
    fig.suptitle("MMLU-Pro: Test Information Curve (5 subjects)", y=1.02)
    fig.savefig(OUTPUT_DIR / "tic.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/tic.png")
    plt.close(fig)

    # --- Plot 7: Wright Map ---
    fig = plot_wright_map(results)
    fig.suptitle("MMLU-Pro: Wright Map", y=1.02)
    fig.savefig(OUTPUT_DIR / "wright_map.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/wright_map.png")
    plt.close(fig)

    # --- Plot 8: Quality pie chart ---
    fig, ax = plt.subplots(figsize=(7, 7))
    quality_data = report["quality"].value_counts()
    colors_pie = {"good": "#4CAF50", "acceptable": "#FFC107", "poor": "#F44336"}
    ax.pie(
        quality_data.values,
        labels=[f"{k}\n({v} items)" for k, v in quality_data.items()],
        colors=[colors_pie.get(k, "gray") for k in quality_data.index],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title("MMLU-Pro Item Quality (IRT Diagnostic)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "quality_pie.png", dpi=150)
    print(f"  Saved: {OUTPUT_DIR}/quality_pie.png")
    plt.close(fig)

    print(f"\nDone! All {len(list(OUTPUT_DIR.glob('*.png')))} plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
