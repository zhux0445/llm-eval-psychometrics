"""End-to-end IRT analysis demo using simulated MMLU-like data.

This script demonstrates the full IRT workflow:
1. Generate MMLU-like benchmark data
2. Fit a 2PL IRT model
3. Inspect item parameters and flag poor items
4. Generate a diagnostic report
5. Visualize ICC, IIC, TIC, and Wright Map
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures

import numpy as np

from llm_eval_psychometrics.irt import (
    IRTAnalyzer,
    classical_item_stats,
    diagnostic_report,
    plot_icc,
    plot_iic,
    plot_tic,
    plot_wright_map,
)
from llm_eval_psychometrics.utils.data import simulate_mmlu_like


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Generate MMLU-like data: 15 models × 100 items (5 subjects × 20 each)
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: Generating MMLU-like benchmark data")
    print("=" * 70)

    responses, model_ids, item_ids, item_subjects = simulate_mmlu_like(
        n_models=15,
        items_per_subject=20,
        seed=42,
    )
    print(f"Response matrix shape: {responses.shape}")
    print(f"Models: {len(model_ids)}, Items: {len(item_ids)}")
    print(f"Subjects: {sorted(set(item_subjects))}")
    print(f"Overall accuracy: {responses.mean():.3f}")
    print()

    # -------------------------------------------------------------------------
    # 2. Fit 2PL IRT model
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 2: Fitting 2PL IRT model")
    print("=" * 70)

    analyzer = IRTAnalyzer(model="2PL")
    results = analyzer.fit(responses, item_ids=item_ids, model_ids=model_ids)

    print("\nItem parameters (first 10):")
    print(results.item_params.head(10).to_string())
    print(f"\nDiscrimination range: [{results.item_params['discrimination'].min():.3f}, "
          f"{results.item_params['discrimination'].max():.3f}]")
    print(f"Difficulty range: [{results.item_params['difficulty'].min():.3f}, "
          f"{results.item_params['difficulty'].max():.3f}]")
    print()

    print("Model ability estimates:")
    for mid, theta in results.model_ability.items():
        print(f"  {mid}: θ = {theta:+.3f}")
    print()

    # -------------------------------------------------------------------------
    # 3. Flag poor items
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 3: Flagging poor items")
    print("=" * 70)

    poor = results.flag_poor_items(min_discrimination=0.3, max_difficulty_abs=3.0)
    if len(poor) > 0:
        print(f"\n{len(poor)} items flagged as poor quality:")
        for idx, row in poor.iterrows():
            print(f"  {idx}: {', '.join(row['reasons'])}")
    else:
        print("\nNo items flagged — all items meet quality thresholds.")
    print()

    # -------------------------------------------------------------------------
    # 4. Full diagnostic report
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 4: Diagnostic report")
    print("=" * 70)

    report = diagnostic_report(results)
    quality_counts = report["quality"].value_counts()
    print(f"\nItem quality distribution:")
    for q in ["good", "acceptable", "poor"]:
        count = quality_counts.get(q, 0)
        print(f"  {q}: {count} items ({count / len(report) * 100:.0f}%)")

    print(f"\nClassical stats summary:")
    print(f"  Mean p-value (easiness): {report['p_value'].mean():.3f}")
    print(f"  Mean point-biserial r: {report['point_biserial'].mean():.3f}")
    print(f"  Mean RMSR (fit): {report['rmsr'].mean():.3f}")
    print()

    # -------------------------------------------------------------------------
    # 5. Visualizations
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 5: Generating plots")
    print("=" * 70)

    # Select a few items for ICC/IIC plots
    sample_items = [item_ids[0], item_ids[20], item_ids[40], item_ids[60], item_ids[80]]

    fig = plot_icc(results, item_ids=sample_items)
    fig.savefig("examples/demo_icc.png", dpi=150)
    print("  Saved: examples/demo_icc.png")

    fig = plot_iic(results, item_ids=sample_items)
    fig.savefig("examples/demo_iic.png", dpi=150)
    print("  Saved: examples/demo_iic.png")

    fig = plot_tic(results, show_se=True)
    fig.savefig("examples/demo_tic.png", dpi=150)
    print("  Saved: examples/demo_tic.png")

    fig = plot_wright_map(results)
    fig.savefig("examples/demo_wright_map.png", dpi=150)
    print("  Saved: examples/demo_wright_map.png")

    print("\nDone! All plots saved to examples/")


if __name__ == "__main__":
    main()
