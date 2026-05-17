# llm-eval-psychometrics

**Psychometric diagnostics for LLM benchmarks and evaluation systems.**

Use Item Response Theory (IRT) and classical statistical methods to diagnose benchmark quality, detect judge bias, and quantify evaluation reliability.

## Why Psychometrics for LLM Evaluation?

LLM benchmarks are standardized tests. LLM-as-judge systems are rating scales. The tools that psychometricians have refined for decades apply directly:

| Problem | Psychometric Solution | Module |
|---------|----------------------|--------|
| Which benchmark items are garbage? | IRT item analysis | `irt` |
| Does my LLM judge have systematic bias? | Bias detection tests | `bias` |
| Can I trust these evaluation results? | Reliability coefficients | `reliability` |
| How many samples do I need? | Power analysis | `reliability` |

## Installation

```bash
pip install llm-eval-psychometrics
```

## Quick Example

```python
import numpy as np
from llm_eval_psychometrics.irt import IRTAnalyzer, diagnostic_report

# 10 LLMs evaluated on 50 benchmark items
responses, true_params = simulate_response_matrix(n_models=10, n_items=50)

analyzer = IRTAnalyzer(model="2PL")
results = analyzer.fit(responses)

# Which items are worth keeping?
report = diagnostic_report(results)
print(report["quality"].value_counts())
```

## Modules

- **[IRT](api/irt.md)** — Fit 2PL/3PL models, flag poor items, plot information curves
- **[Bias Detection](api/bias.md)** — Position bias, length bias, self-preference, score calibration
- **[Reliability](api/reliability.md)** — Cohen's Kappa, ICC, Krippendorff's alpha, bootstrap CI, power analysis
- **[Utilities](api/utils.md)** — Synthetic data generators for testing

## Methodological Note: Calibration Sample Size

!!! warning "Item parameter estimates require an adequate model pool"

    Classical IRT calibration assumes many examinees relative to items
    (typically hundreds to thousands). When the toolkit is applied to a
    small model pool — e.g., the 10–15 model demos in this documentation —
    **item parameter estimates are provisional** and carry large standard
    errors.

    Large-scale, production-grade benchmark calibration would require
    expanding to **~hundreds of models**, as in
    [tinyBenchmarks (Polo et al., 2024)](https://arxiv.org/abs/2402.14992),
    which calibrates item parameters on the Open LLM Leaderboard's several
    hundred models before freezing those parameters for evaluating any new
    model.

    Interpret the demos in this documentation as proof-of-concept of the
    analysis pipeline, not as precise population estimates of item quality.

## Next Steps

- [Quickstart guide](quickstart.md) with copy-paste examples
- [IRT demo](examples/irt_demo.md) with end-to-end walkthrough
- [API reference](api/irt.md) for full documentation
