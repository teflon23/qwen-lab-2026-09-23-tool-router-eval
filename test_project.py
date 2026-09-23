import numpy as np
import pytest

from data import make_dataset
from app import run_experiment


def test_invalid_n_samples_raises_value_error():
    """Edge case: n_samples < 32 must raise ValueError."""
    with pytest.raises(ValueError):
        make_dataset(seed=42, n_samples=31)
    with pytest.raises(ValueError):
        make_dataset(seed=0, n_samples=0)
    with pytest.raises(ValueError):
        make_dataset(seed=0, n_samples=1)


def test_different_seeds_produce_different_data():
    """Different seeds must change X."""
    X1, _ = make_dataset(seed=42, n_samples=100)
    X2, _ = make_dataset(seed=99, n_samples=100)
    assert not np.array_equal(X1, X2)


def test_learned_router_accuracy_exceeds_threshold():
    """TF-IDF + Logistic Regression accuracy on held-out test must exceed 0.85."""
    result = run_experiment(seed=42, n_samples=256)
    assert result["metrics"]["accuracy"] > 0.85


def test_learned_router_macro_f1_exceeds_baseline():
    """Macro F1 of learned router must strictly exceed keyword baseline."""
    result = run_experiment(seed=42, n_samples=256)
    assert result["metrics"]["macro_f1"] > result["baseline_metrics"]["macro_f1"]


def test_n_samples_is_respected():
    """Different n_samples must be reflected in the output dict."""
    r1 = run_experiment(seed=42, n_samples=100)
    r2 = run_experiment(seed=42, n_samples=200)
    assert r1["n_samples"] == 100
    assert r2["n_samples"] == 200


def test_deterministic_output():
    """Same inputs must give exactly the same output dict."""
    r1 = run_experiment(seed=42, n_samples=256)
    r2 = run_experiment(seed=42, n_samples=256)
    assert r1 == r2


def test_dataset_shape_and_valid_labels():
    """Dataset must have correct length and labels in {0..4}."""
    X, y = make_dataset(seed=7, n_samples=64)
    assert len(X) == 64
    assert len(y) == 64
    assert set(np.unique(y).tolist()) == {0, 1, 2, 3, 4}
    assert all(isinstance(s, str) for s in X)
