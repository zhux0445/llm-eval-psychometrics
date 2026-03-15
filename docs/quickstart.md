# Quickstart

## Installation

```bash
pip install llm-eval-psychometrics
```

## IRT Analysis

```python
import numpy as np
from llm_eval_psychometrics.irt import IRTAnalyzer

# Binary response matrix: rows = models, columns = benchmark items
response_matrix = np.array([...])  # shape: (n_models, n_items)

analyzer = IRTAnalyzer(model="2PL")
results = analyzer.fit(response_matrix)

# Inspect item parameters
print(results.item_params)

# Flag low-quality items
poor_items = results.flag_poor_items()
print(f"Items to review: {poor_items}")
```

## Bias Detection

```python
import pandas as pd
from llm_eval_psychometrics.bias import BiasDetector

scores_df = pd.read_csv("judge_scores.csv")
detector = BiasDetector(scores_df)

print(detector.position_bias())
print(detector.length_bias())
detector.report()
```

## Reliability

```python
from llm_eval_psychometrics.reliability import AgreementAnalyzer, PowerCalculator

agree = AgreementAnalyzer(ratings_df)
print(f"Kappa: {agree.cohens_kappa()}")
print(f"ICC: {agree.icc()}")

power = PowerCalculator()
n = power.min_items_for_precision(target_se=0.1)
print(f"Minimum items needed: {n}")
```
