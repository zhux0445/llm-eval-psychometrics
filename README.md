# llm-eval-psychometrics

[![PyPI version](https://img.shields.io/pypi/v/llm-eval-psychometrics.svg)](https://pypi.org/project/llm-eval-psychometrics/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-98%20passed-brightgreen.svg)]()

**Psychometric diagnostics for LLM benchmarks and evaluation systems.**

Use Item Response Theory (IRT) and classical statistical methods to answer:

- Is this benchmark actually measuring what we think? Which items are garbage?
- Does this LLM-as-judge have systematic biases (position, length, self-preference)?
- How reliable are these evaluation results? How many samples do we actually need?

## Installation

```bash
pip install llm-eval-psychometrics
```

For development:

```bash
git clone https://github.com/ruoyizhu/llm-eval-psychometrics.git
cd llm-eval-psychometrics
pip install -e ".[dev]"
```

## Quickstart

### IRT Analysis — Diagnose Benchmark Quality

```python
import numpy as np
from llm_eval_psychometrics.irt import IRTAnalyzer, diagnostic_report

# Binary response matrix: rows = LLMs, columns = benchmark items
# 1 = correct, 0 = incorrect
response_matrix = np.array([
    [1, 1, 0, 1, 0, 1, 1, 0, 0, 1],  # GPT-4
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # LLaMA-70B
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],  # Claude-3
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Mistral-7B
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],  # Gemini
])

analyzer = IRTAnalyzer(model="2PL")
results = analyzer.fit(
    response_matrix,
    model_ids=["GPT-4", "LLaMA-70B", "Claude-3", "Mistral-7B", "Gemini"],
)

# Item parameters: discrimination (how well it separates models) & difficulty
print(results.item_params)

# Model ability estimates (θ) on a common scale
print(results.model_ability)

# Flag low-quality items (low discrimination or extreme difficulty)
poor_items = results.flag_poor_items()
print(f"Items to review: {len(poor_items)}")

# Full diagnostic report with quality classification
report = diagnostic_report(results)
print(report[["discrimination", "difficulty", "p_value", "quality"]])
```

### Bias Detection — Audit LLM-as-Judge Systems

```python
import pandas as pd
from llm_eval_psychometrics.bias import BiasDetector

scores_df = pd.DataFrame({
    "score": [4, 3, 5, 2, 4, 3, 5, 2, 3, 4],
    "position": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    "response_length": [150, 80, 300, 50, 200, 90, 400, 60, 100, 250],
    "judge_model": ["gpt-4"] * 5 + ["claude-3"] * 5,
    "response_model": ["gpt-3.5", "claude-2", "llama-70b", "gpt-3.5", "claude-2",
                        "claude-2", "gpt-3.5", "llama-70b", "claude-2", "gpt-3.5"],
})

detector = BiasDetector(scores_df)

# Test individual biases
print(detector.position_bias())    # Are scores affected by option order?
print(detector.length_bias())      # Do longer responses get higher scores?
print(detector.self_preference())  # Do judges favor their own model family?

# Or run all applicable tests at once
print(detector.report())
```

### Score Calibration — Make Judges Comparable

```python
from llm_eval_psychometrics.bias import ScoreCalibrator

# When different LLM judges use different parts of the scale
calibrator = ScoreCalibrator(scores_df)
result = calibrator.z_score_calibration()  # or percentile_calibration()
print(result.calibrated_scores[["judge_model", "score", "calibrated_score"]])
```

### Reliability — Measure Agreement & Plan Sample Sizes

```python
import numpy as np
from llm_eval_psychometrics.reliability import AgreementAnalyzer, PowerCalculator

# Inter-rater agreement between LLM judges
ratings = pd.DataFrame({
    "judge_A": [4, 3, 5, 2, 4, 3],
    "judge_B": [3, 3, 4, 2, 5, 3],
    "judge_C": [4, 2, 5, 2, 4, 3],
})

agree = AgreementAnalyzer(ratings)
print(agree.cohens_kappa(weighted="quadratic"))
print(agree.icc(model="two-way-mixed"))
print(agree.krippendorff_alpha())

# Sample size planning
power = PowerCalculator()

# How many benchmark items for precise ability estimates?
print(power.min_items_for_precision(target_se=0.1))

# How many samples to detect a difference between two models?
print(power.min_samples_for_difference(effect_size=0.5, power=0.80))

# Bootstrap confidence interval on evaluation scores
scores = np.array([0.82, 0.79, 0.85, 0.81, 0.83, 0.78, 0.84])
print(power.bootstrap_ci(scores, n_bootstrap=2000))
```

### Visualization

```python
from llm_eval_psychometrics.irt import plot_icc, plot_iic, plot_tic, plot_wright_map

# Item Characteristic Curves — probability of correct response vs ability
fig = plot_icc(results, item_ids=["item_0", "item_3", "item_7"])

# Item Information Curves — where each item measures best
fig = plot_iic(results)

# Test Information Curve — overall measurement precision across ability range
fig = plot_tic(results, show_se=True)

# Wright Map — are items well-targeted to model ability levels?
fig = plot_wright_map(results)
```

### Demand Profiling — Multidimensional Benchmark Analysis

Inspired by Zhou et al. (2026) "General scales unlock AI evaluation with explanatory and predictive power" (*Nature*, Vol 652), the `demand` module lets you annotate benchmark items across cognitive demand dimensions and answer: *what does this benchmark actually measure, and how well does each model handle each type of demand?*

```python
import numpy as np
import pandas as pd
from llm_eval_psychometrics.demand import DemandProfiler

# Demand matrix: rows = items, columns = cognitive demand dimensions
# Values are demand levels (e.g. 0–5)
demand_matrix = pd.DataFrame({
    "logical_reasoning":      [3, 5, 2, 4, 1],
    "quantitative_reasoning": [4, 5, 1, 3, 2],
    "verbal_comprehension":   [1, 0, 2, 1, 0],
    "knowledge_formal_sciences": [2, 3, 0, 4, 1],
    "atypicality":            [0, 1, 0, 2, 0],
}, index=["item_0", "item_1", "item_2", "item_3", "item_4"])

# Binary response matrix: rows = models, columns = items
response_matrix = np.array([
    [1, 1, 1, 0, 1],  # GPT-4o
    [1, 0, 1, 0, 1],  # Claude-3
    [0, 0, 1, 0, 1],  # LLaMA-3-70B
])

profiler = DemandProfiler(
    demand_matrix=demand_matrix,
    response_matrix=response_matrix,
    model_ids=["GPT-4o", "Claude-3", "LLaMA-3-70B"],
)

# Demand profile: what does the benchmark measure?
profile = profiler.demand_profile()
print(profile.summary())

# Construct validity: is the benchmark sensitive to target dimensions?
validity = profiler.benchmark_validity(
    benchmark_name="Math Benchmark",
    target_dimensions=["logical_reasoning", "quantitative_reasoning"],
)
print(f"Sensitivity: {validity.overall_sensitivity:.2f}")
print(f"Specificity: {validity.overall_specificity:.2f}")

# Per-dimension ability profiles for each model
abilities = profiler.ability_profile()
print(abilities)

# Predict instance-level performance from demand-ability gaps
predicted = profiler.predict_performance(ability_profiles=abilities)
```

## Modules

| Module | Purpose | Key Classes |
|--------|---------|------------|
| `irt` | Benchmark item analysis via IRT | `IRTAnalyzer`, `IRTResults` |
| `bias` | LLM judge bias detection & calibration | `BiasDetector`, `ScoreCalibrator` |
| `reliability` | Inter-rater agreement & power analysis | `AgreementAnalyzer`, `PowerCalculator` |
| `demand` | Multidimensional demand profiling & construct validity | `DemandProfiler`, `DemandProfile`, `BenchmarkValidity` |
| `utils` | Data generation & loading utilities | `simulate_mmlu_like()` |

## Key Concepts

**Why psychometrics for LLM evaluation?**

LLM benchmarks are essentially standardized tests, and LLM-as-judge systems are rating scales — the same tools psychometricians have refined for decades apply directly:

- **IRT** tells you which benchmark items actually differentiate between models (high discrimination) vs. items every model gets right/wrong (uninformative) or items that behave randomly (noisy).
- **Demand profiling** (Zhou et al., 2026) goes beyond overall accuracy by annotating items across cognitive demand dimensions — quantifying what a benchmark truly measures, assessing construct validity, and estimating per-dimension ability profiles that explain *why* one model outperforms another.
- **Bias detection** catches when your LLM judge systematically favors longer responses, responses in certain positions, or responses from its own model family.
- **Reliability analysis** quantifies how much you can trust your evaluation results and tells you exactly how many samples you need.

## Contributing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
black --check .
```

## License

MIT
