# IRT Analysis Demo

End-to-end example: analyze a simulated MMLU-like benchmark with IRT.

## Generate Data

```python
from llm_eval_psychometrics.utils.data import simulate_mmlu_like

responses, model_ids, item_ids, item_subjects = simulate_mmlu_like(
    n_models=15,
    items_per_subject=20,
    seed=42,
)
print(f"Response matrix: {responses.shape}")
# Response matrix: (15, 100)
```

## Fit 2PL Model

```python
from llm_eval_psychometrics.irt import IRTAnalyzer

analyzer = IRTAnalyzer(model="2PL")
results = analyzer.fit(responses, item_ids=item_ids, model_ids=model_ids)

print(results.item_params.head())
print(results.model_ability)
```

## Flag Poor Items

```python
poor = results.flag_poor_items(min_discrimination=0.3)
print(f"{len(poor)} items flagged:")
for idx, row in poor.iterrows():
    print(f"  {idx}: {', '.join(row['reasons'])}")
```

## Diagnostic Report

```python
from llm_eval_psychometrics.irt import diagnostic_report

report = diagnostic_report(results)
print(report["quality"].value_counts())
# good          23
# acceptable    58
# poor          19
```

## Visualize

```python
from llm_eval_psychometrics.irt import plot_icc, plot_iic, plot_tic, plot_wright_map

# Item Characteristic Curves for selected items
fig = plot_icc(results, item_ids=item_ids[:5])
fig.savefig("icc.png", dpi=150)

# Item Information Curves
fig = plot_iic(results, item_ids=item_ids[:5])
fig.savefig("iic.png", dpi=150)

# Test Information Curve with Standard Error
fig = plot_tic(results, show_se=True)
fig.savefig("tic.png", dpi=150)

# Wright Map
fig = plot_wright_map(results)
fig.savefig("wright_map.png", dpi=150)
```

### Sample Output: Test Information Curve

The TIC shows where your benchmark measures most precisely. High information (blue) means low standard error (red dashed). If the peak doesn't align with your models' ability range, the benchmark is poorly targeted.

### Sample Output: Wright Map

The Wright Map places model abilities (left histogram) and item difficulties (right diamonds) on the same scale. Good benchmarks have items spread across the ability range. Outlier items far above or below all models provide no useful information.

## Run the Full Demo

```bash
python examples/demo_irt.py
```

This generates four plots in the `examples/` directory:

- `demo_icc.png` — Item Characteristic Curves
- `demo_iic.png` — Item Information Curves
- `demo_tic.png` — Test Information Curve
- `demo_wright_map.png` — Wright Map
