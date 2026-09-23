"""Deterministic tool-router evaluation: TF-IDF + Logistic Regression vs keyword baseline."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from data import make_dataset, TOOL_KEYWORDS, N_TOOLS


def _keyword_route(text: str) -> int:
    """Return the first tool index whose keyword appears in the lowercased text.

    Iterates tools in order 0..4; within each tool, checks keywords in list
    order. This mimics a naive prompt-based router that checks a fixed
    keyword checklist and picks the first hit. Falls back to tool 0 if no
    keyword matches (should be rare with the synthetic templates).
    """
    text_lower = text.lower()
    for tool_id in range(N_TOOLS):
        for kw in TOOL_KEYWORDS[tool_id]:
            if kw in text_lower:
                return tool_id
    return 0


def run_experiment(seed=42, n_samples=256) -> dict:
    """Evaluate a learned tool router against a keyword-overlap baseline.

    Data: i.i.d. synthetic agent-task strings from make_dataset.
    Split: stratified 80/20 (random_state=0) so every tool class is
    proportionally represented in both folds; because data is i.i.d.
    (no temporal ordering), a random stratified split avoids both
    class-imbalance and temporal leakage.
    Model: TF-IDF (1-2 grams) + Logistic Regression (liblinear, C=1).
    Baseline: deterministic first-keyword-match over per-tool keyword lists.
    Metrics: accuracy and macro F1 on the held-out test set, plus per-tool F1.

    Returns a JSON-serializable dict.
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    # --- Learned router ---
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(solver="liblinear", max_iter=2000, random_state=0)
    clf.fit(X_train_tfidf, y_train)
    y_pred_learned = clf.predict(X_test_tfidf)

    learned_acc = float(accuracy_score(y_test, y_pred_learned))
    learned_macro_f1 = float(f1_score(y_test, y_pred_learned, average="macro"))
    per_tool_f1 = f1_score(y_test, y_pred_learned, average=None)

    # --- Keyword-overlap baseline ---
    y_pred_baseline = np.array([_keyword_route(t) for t in X_test])
    baseline_acc = float(accuracy_score(y_test, y_pred_baseline))
    baseline_macro_f1 = float(f1_score(y_test, y_pred_baseline, average="macro"))

    metrics: dict[str, float] = {
        "accuracy": learned_acc,
        "macro_f1": learned_macro_f1,
    }
    for i in range(N_TOOLS):
        metrics[f"tool_{i}_f1"] = float(per_tool_f1[i])

    baseline_metrics: dict[str, float] = {
        "accuracy": baseline_acc,
        "macro_f1": baseline_macro_f1,
    }

    explanation = (
        f"Learned router (TF-IDF 1-2gram + LogisticRegression liblinear): "
        f"acc={learned_acc:.4f}, macro-F1={learned_macro_f1:.4f}. "
        f"Keyword baseline (first-match over {N_TOOLS} ordered tool lists): "
        f"acc={baseline_acc:.4f}, macro-F1={baseline_macro_f1:.4f}. "
        f"Stratified 80/20 split: train={len(X_train)}, test={len(X_test)}. "
        f"Data is i.i.d. synthetic (no temporal sequence); stratification "
        f"guarantees each tool class appears in both folds, preventing "
        f"class-imbalance leakage. Per-tool F1 values expose individual "
        f"routing blind spots of the learned classifier."
    )

    return {
        "n_samples": int(n_samples),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "explanation": explanation,
    }
