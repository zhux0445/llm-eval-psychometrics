# llm-eval-psychometrics

Psychometric diagnostics toolkit for LLM benchmarks and evaluation systems. Published on PyPI.

## Build & Install

```bash
pip install -e ".[dev]"        # editable install with dev dependencies
pip install -e ".[dev,docs]"   # include docs dependencies
```

## Testing

```bash
pytest                  # run all tests (uses tests/ directory)
pytest tests/test_irt.py       # run a specific test file
pytest -x              # stop on first failure
```

- Tests are in `tests/` and follow the naming pattern `test_<module>.py`.
- Run tests after making changes to verify nothing breaks.

## Linting & Formatting

```bash
ruff check .           # lint
ruff check --fix .     # lint with auto-fix
black .                # format
```

- Line length: 88 characters (configured in pyproject.toml).

## Project Structure

```
llm_eval_psychometrics/       # main package
├── irt/                      # Item Response Theory models (1PL/2PL/3PL)
├── bias/                     # DIF and bias detection for LLM-as-judge
├── reliability/              # reliability metrics (Cronbach's alpha, etc.)
├── demand/                   # multidimensional demand profiling
└── utils/                    # shared utilities

tests/                        # unit tests
examples/                     # usage demos and notebooks
docs/                         # mkdocs documentation source
```

## Documentation

```bash
mkdocs serve           # local docs preview at http://127.0.0.1:8000
mkdocs build           # build static site to site/
```

## Key Dependencies

- `numpy`, `scipy`, `pandas` — numerical computation
- `girth` — IRT parameter estimation
- `krippendorff` — inter-rater reliability
- `matplotlib` — visualization

## Git

- Main branch: `main`
- Repo: github.com/zhux0445/llm-eval-psychometrics
