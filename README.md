# Deterministic Tool-Router Evaluation for Synthetic Agent Tasks

## Problem

In multi-tool agent systems, the router must decide which tool to invoke for a given user request. A common production shortcut is a keyword checklist or a simple prompt-based heuristic. This project evaluates whether a small, deterministic, learned classifier (TF-IDF + Logistic Regression) can route synthetic agent-task descriptions more accurately than a naive first-keyword-match baseline, and exposes per-tool routing blind spots via per-class F1.

## What This Project Does

- Generates 256 (default) synthetic natural-language agent-task strings across 5 tools (search, calculator, weather, calendar, email) using seeded template fill-in.
- Splits the data with a stratified 80/20 random split (i.i.d. data, no temporal ordering).
- Trains a TF-IDF (1–2 grams) + Logistic Regression (liblinear) classifier on the training fold.
- Evaluates both the learned router and a deterministic keyword-overlap baseline on the held-out test fold.
- Reports accuracy, macro F1, and per-tool F1 for the learned router; accuracy and macro F1 for the baseline.

This is a **deterministic simulation** of a routing decision. No LLM is called, no network is used, and no external data is loaded.

## Architecture

```
data.py          → make_dataset(seed, n_samples) → (X: str array, y: int array)
app.py           → run_experiment(seed, n_samples) → JSON-serializable dict
main.py          → CLI entry point (host-supplied)
test_project.py  → pytest tests
```

- `data.py` defines 5 tool templates with verb/object/day/person pools and a `TOOL_KEYWORDS` dict used by the baseline.
- `app.py` contains the TF-IDF + Logistic Regression pipeline and the `_keyword_route` baseline function.
- `main.py` (host-supplied) calls `run_experiment` and writes the result to a JSON file.

## Synthetic Dataset Assumptions

- **i.i.d. generation**: Each sample is drawn independently; there is no temporal sequence, so a random stratified split is appropriate and no chronological split is needed.
- **Balanced classes**: Tool index is assigned round-robin (`i % 5`), giving exactly 20% per tool in the full dataset. Stratification preserves this balance in both folds.
- **Template-based text**: Strings are generated from 5 patterns per tool with random fill-in from small pools. This creates realistic-looking but constrained vocabulary. The limited vocabulary means the learned classifier has a structural advantage over the keyword baseline, which is intentional for demonstrating the gap.
- **No noise or ambiguity**: Each string is unambiguously associated with one tool. Real-world routing involves overlapping intent, misspellings, and multi-tool requests, which this dataset does not model.
- **Limitations**: The dataset is small, synthetic, and lacks the linguistic diversity, ambiguity, and distribution shift of real agent traffic. Results here do not predict production routing performance.

## Algorithm vs. Baseline

| Component | Learned Router | Keyword Baseline |
|---|---|---|
| Feature extraction | TF-IDF, 1–2 grams, lowercase | None (raw string matching) |
| Decision rule | Logistic Regression (liblinear, C=1, max_iter=2000) | First keyword match in ordered tool list (0→4); fallback to tool 0 |
| Training | Fit on 80% stratified train split | No training; static keyword lists |
| Interpretability | Coefficients inspectable | Fully transparent rule |

The keyword baseline mimics a naive prompt-based router that checks a fixed checklist and picks the first hit. The learned router is a standard text-classification pipeline.

## Metrics and Direction

All metrics are computed on the **held-out test set** only.

| Metric | Direction | Meaning |
|---|---|---|
| `accuracy` | Higher is better | Fraction of correctly routed samples |
| `macro_f1` | Higher is better | Mean of per-tool F1; treats all tools equally |
| `tool_{i}_f1` (i=0..4) | Higher is better | Per-tool F1; exposes individual routing blind spots |

The learned router is expected to exceed the keyword baseline on both accuracy and macro F1. Per-tool F1 values reveal which specific tools the learned classifier still struggles with.

## Reproducibility

- **Python**: 3.11, 3.12, or 3.13.
- **Dependencies**: `numpy`, `scikit-learn`, `pytest` (versions pinned in `requirements.txt`).
- **Determinism**: All randomness is seeded (`make_dataset(seed)`, `train_test_split(random_state=0)`, `LogisticRegression(random_state=0)`). Same inputs produce identical outputs.
- **No network, no disk reads, no external data.**

### Setup and Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --seed 42 --n-samples 256 --output results.json
python -m pytest -q
```

### Output

`main.py` writes a JSON file (e.g., `results.json`) containing:

- `n_samples`: int
- `metrics`: dict with `accuracy`, `macro_f1`, `tool_0_f1` … `tool_4_f1`
- `baseline_metrics`: dict with `accuracy`, `macro_f1`
- `explanation`: human-readable summary string

See `example_results.json` for a sample output structure and `validation_report.json` for measured values from the acceptance test run.

## Acceptance Criteria

The project is considered correct when:

1. TF-IDF + Logistic Regression routing accuracy on the held-out test set exceeds 0.85.
2. Macro F1 of the learned router strictly exceeds macro F1 of the keyword-overlap baseline on the same test split.
3. `make_dataset` raises `ValueError` for `n_samples=31` and returns different `X` arrays for different seeds.

These are verified by `test_project.py` and the host-supplied `main.py` contract tests.

## Limitations and Honesty Disclosure

- **Synthetic data only**: The dataset is generated from templates and does not represent real user language, ambiguity, or multi-tool requests.
- **Small scale**: 256 samples is sufficient for a demonstration but not for statistical significance claims.
- **No LLM involved**: This project does not call, simulate, or approximate any LLM. It is a classical ML text-classification evaluation.
- **No production readiness**: This is an educational, reproducible demo. It is not a production routing system.
- **Automated tests do not prove correctness**: Passing tests verify interface contracts and basic behavioral properties, not full semantic correctness of the routing logic.
- **No CI, no hosted service, no badges**: There is no continuous integration workflow, no hosted API, and no external URLs.

## Provenance

This project was generated by Qwen. All code is original and uses only synthetic data. No third-party datasets, licenses, or external resources are used.

## Recorded automated validation

Host contract tests and project tests passed (11 tests, 0 skipped). Demo completed on Python 3.13.15. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
