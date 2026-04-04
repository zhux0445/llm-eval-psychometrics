"""End-to-end demand profiling demo using simulated benchmark data.

This script demonstrates the full demand profiling workflow:
1. Create a simulated benchmark with multidimensional demand annotations
2. Create a simulated response matrix for 8 LLMs
3. Compute a demand profile and inspect its summary
4. Assess benchmark validity (sensitivity & specificity)
5. Estimate per-dimension ability profiles for each model
6. Predict performance from demand-ability gaps and compare to actual
7. Compute characteristic curves per demand dimension
8. Compute inter-dimension correlations
9. Generate and save all three diagnostic plots
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures

import numpy as np
import pandas as pd

from llm_eval_psychometrics.demand import DemandProfiler
from llm_eval_psychometrics.demand.plots import (
    plot_demand_profile,
    plot_ability_vs_demand,
    plot_sensitivity_specificity,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


def simulate_demand_benchmark(
    n_items: int = 80,
) -> pd.DataFrame:
    """Simulate demand annotations for a math-heavy benchmark.

    The benchmark is designed to lean heavily on logical_reasoning and
    quantitative_reasoning, with moderate knowledge_formal_sciences
    and small amounts of verbal_comprehension and atypicality.
    """
    dims = [
        "logical_reasoning",
        "quantitative_reasoning",
        "verbal_comprehension",
        "knowledge_formal_sciences",
        "knowledge_natural_sciences",
        "atypicality",
        "volume",
    ]

    data = {}

    # Target dimensions: high and varied
    data["logical_reasoning"] = RNG.integers(1, 6, size=n_items).astype(float)
    data["quantitative_reasoning"] = RNG.integers(1, 6, size=n_items).astype(float)

    # Moderate presence
    data["knowledge_formal_sciences"] = RNG.integers(0, 4, size=n_items).astype(float)

    # Low presence — extraneous
    data["verbal_comprehension"] = RNG.integers(0, 3, size=n_items).astype(float)
    data["knowledge_natural_sciences"] = RNG.integers(0, 2, size=n_items).astype(float)

    # Very low — extraneous noise
    data["atypicality"] = RNG.choice(
        [0, 1, 2], size=n_items, p=[0.6, 0.3, 0.1]
    ).astype(float)
    data["volume"] = RNG.choice(
        [0, 1, 2, 3], size=n_items, p=[0.4, 0.35, 0.2, 0.05]
    ).astype(float)

    item_ids = [f"item_{i:03d}" for i in range(n_items)]
    return pd.DataFrame(data, index=item_ids)


def simulate_responses(
    demand_matrix: pd.DataFrame,
    model_ids: list[str],
) -> np.ndarray:
    """Simulate binary response matrix (n_models x n_items).

    Each model has a latent ability vector across dimensions.  The
    probability of a correct response is a logistic function of the
    mean (ability - demand) gap across all dimensions.
    """
    n_items = len(demand_matrix)
    n_models = len(model_ids)
    dims = list(demand_matrix.columns)

    # Ground-truth ability levels per model, per dimension (0-5 scale)
    ability_matrix = np.array([
        # log  quant  verbal  k_form  k_nat  atyp  vol
        [4.5,  4.8,   3.5,    4.2,   3.0,   3.5,  3.0],  # GPT-4o
        [4.2,  4.0,   4.0,    3.8,   3.5,   2.5,  3.0],  # Claude-3-Opus
        [3.8,  3.5,   3.8,    3.2,   3.0,   2.0,  2.5],  # Gemini-1.5-Pro
        [3.5,  3.8,   3.0,    3.5,   2.8,   2.0,  2.5],  # LLaMA-3-70B
        [3.0,  3.2,   3.2,    3.0,   2.5,   1.5,  2.0],  # Mixtral-8x22B
        [2.5,  2.8,   2.8,    2.5,   2.2,   1.5,  2.0],  # LLaMA-3-8B
        [2.0,  2.2,   2.5,    2.0,   2.0,   1.0,  1.5],  # Mistral-7B
        [1.5,  1.8,   2.0,    1.5,   1.5,   1.0,  1.5],  # Phi-3-Mini
    ])

    responses = np.zeros((n_models, n_items))
    demand_arr = demand_matrix[dims].values  # (n_items, n_dims)

    for m in range(n_models):
        for i in range(n_items):
            gaps = ability_matrix[m] - demand_arr[i]
            avg_gap = gaps.mean()
            prob = 1.0 / (1.0 + np.exp(-avg_gap))
            responses[m, i] = RNG.binomial(1, prob)

    return responses


def main() -> None:
    model_ids = [
        "GPT-4o",
        "Claude-3-Opus",
        "Gemini-1.5-Pro",
        "LLaMA-3-70B",
        "Mixtral-8x22B",
        "LLaMA-3-8B",
        "Mistral-7B",
        "Phi-3-Mini",
    ]

    # -------------------------------------------------------------------------
    # 1. Simulate benchmark
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: Simulating benchmark demand annotations")
    print("=" * 70)

    demand_matrix = simulate_demand_benchmark(n_items=80)
    responses = simulate_responses(demand_matrix, model_ids)

    print(f"Demand matrix shape:   {demand_matrix.shape}  (items × dimensions)")
    print(f"Response matrix shape: {responses.shape}  (models × items)")
    print(f"Dimensions: {list(demand_matrix.columns)}")
    print(f"Overall accuracy: {responses.mean():.3f}")
    print()

    # -------------------------------------------------------------------------
    # 2. Initialise profiler
    # -------------------------------------------------------------------------
    profiler = DemandProfiler(
        demand_matrix=demand_matrix,
        response_matrix=responses,
        model_ids=model_ids,
    )

    # -------------------------------------------------------------------------
    # 3. Demand profile
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 2: Demand profile — what does this benchmark measure?")
    print("=" * 70)

    profile = profiler.demand_profile()
    summary = profile.summary()
    print("\nDemand summary (per dimension):")
    print(summary.to_string(float_format="{:.3f}".format))
    print()

    # -------------------------------------------------------------------------
    # 4. Benchmark validity
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 3: Benchmark validity for 'Math Reasoning' benchmark")
    print("=" * 70)

    validity = profiler.benchmark_validity(
        benchmark_name="Math Reasoning Benchmark",
        target_dimensions=["logical_reasoning", "quantitative_reasoning",
                           "knowledge_formal_sciences"],
        min_levels=3,
        max_extraneous_mean=0.5,
    )

    print(f"\nOverall sensitivity:  {validity.overall_sensitivity:.3f}")
    print(f"Overall specificity:  {validity.overall_specificity:.3f}")
    print("\nSensitivity by target dimension:")
    for _, row in validity.sensitivity.iterrows():
        bar = "#" * int(row["sensitivity"] * 20)
        print(f"  {row['dimension']:<30s} {row['sensitivity']:.3f}  [{bar:<20s}]  {row['detail']}")
    print("\nSpecificity by extraneous dimension (top 4):")
    spec_top = validity.specificity.sort_values("specificity").head(4)
    for _, row in spec_top.iterrows():
        bar = "#" * int(row["specificity"] * 20)
        print(f"  {row['dimension']:<30s} {row['specificity']:.3f}  [{bar:<20s}]  {row['detail']}")
    print()

    # -------------------------------------------------------------------------
    # 5. Ability profiles
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 4: Per-dimension ability profiles for each model")
    print("=" * 70)

    abilities = profiler.ability_profile()
    print("\nAbility estimates (rows = models, cols = dimensions):")
    print(abilities.to_string(float_format="{:.2f}".format))
    print()

    # Overall ranking by mean ability (excluding NaN)
    mean_ability = abilities.mean(axis=1).sort_values(ascending=False)
    print("Model ranking by mean ability:")
    for rank, (model, val) in enumerate(mean_ability.items(), 1):
        print(f"  {rank}. {model:<20s}  mean ability = {val:.3f}")
    print()

    # -------------------------------------------------------------------------
    # 6. Predict performance and compare to actual
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 5: Predicted vs actual performance")
    print("=" * 70)

    predicted = profiler.predict_performance(ability_profiles=abilities)
    actual_acc = responses.mean(axis=1)  # per-model accuracy

    print(f"\n{'Model':<20s}  {'Actual Acc':>10s}  {'Predicted Acc':>13s}  {'Error':>8s}")
    print("-" * 58)
    for i, model_id in enumerate(model_ids):
        pred_acc = predicted.loc[model_id].mean()
        err = pred_acc - actual_acc[i]
        print(f"  {model_id:<18s}  {actual_acc[i]:>10.3f}  {pred_acc:>13.3f}  {err:>+8.3f}")
    print()

    # -------------------------------------------------------------------------
    # 7. Characteristic curves
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 6: Characteristic curves (success rate vs demand level)")
    print("=" * 70)

    curves = profiler.characteristic_curves(n_bins=5)
    print(f"\nCharacteristic curves computed for {len(curves)} dimensions.")

    for dim in ["logical_reasoning", "quantitative_reasoning"]:
        if dim not in curves:
            continue
        df = curves[dim]
        print(f"\n  Dimension: {dim}")
        print(f"  {'Demand':>8s}", end="")
        for m in model_ids[:4]:   # show first 4 models for brevity
            print(f"  {m[:10]:>10s}", end="")
        print()
        for _, row in df.iterrows():
            print(f"  {row['demand_level']:>8.2f}", end="")
            for m in model_ids[:4]:
                val = row[m]
                display = f"{val:.3f}" if not np.isnan(val) else "   nan"
                print(f"  {display:>10s}", end="")
            print()
    print()

    # -------------------------------------------------------------------------
    # 8. Dimension correlations
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 7: Spearman correlations between demand dimensions")
    print("=" * 70)

    corr = profiler.dimension_correlations()
    print("\nCorrelation matrix:")
    print(corr.to_string(float_format="{:.3f}".format))

    # Identify high correlations (|r| > 0.5, off-diagonal)
    high_corr = []
    dims = list(corr.columns)
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            if j <= i:
                continue
            r = corr.loc[d1, d2]
            if abs(r) > 0.5:
                high_corr.append((d1, d2, r))
    if high_corr:
        print("\nDimension pairs with |r| > 0.5 (potential overlap):")
        for d1, d2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"  {d1} <-> {d2}: r = {r:.3f}")
    else:
        print("\nNo dimension pairs exceed |r| = 0.5 — dimensions are well separated.")
    print()

    # -------------------------------------------------------------------------
    # 9. Visualizations
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("Step 8: Generating plots")
    print("=" * 70)

    import matplotlib.pyplot as plt

    fig = plot_demand_profile(profile, title="Math Reasoning Benchmark — Demand Profile")
    fig.savefig("examples/demo_demand_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: examples/demo_demand_profile.png")

    fig = plot_ability_vs_demand(
        ability_profiles=abilities,
        demand_profile=profile,
        model_ids=["GPT-4o", "Claude-3-Opus", "LLaMA-3-70B", "Phi-3-Mini"],
    )
    fig.savefig("examples/demo_demand_ability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: examples/demo_demand_ability.png")

    fig = plot_sensitivity_specificity(validity)
    fig.savefig("examples/demo_demand_validity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: examples/demo_demand_validity.png")

    print("\nDone! All plots saved to examples/")


if __name__ == "__main__":
    main()
