"""Explicit metrics and probability validation, independent of model.score()."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate(y_true, probabilities, classes):
    y_true, p, classes = np.asarray(y_true), np.asarray(probabilities), np.asarray(classes)
    if p.shape != (len(y_true), len(classes)):
        raise ValueError(f"Probability shape {p.shape} != {(len(y_true), len(classes))}")
    if not np.isfinite(p).all() or (p < -1e-6).any() or (p > 1 + 1e-6).any():
        raise ValueError("Model returned invalid probabilities")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Probabilities do not sum to one")
    if not np.isin(y_true, classes).all():
        raise ValueError("Test contains a class absent from the fitted model")
    pred = classes[p.argmax(axis=1)]
    # sklearn default label union matches the ordinary macro metric definition;
    # do not silently score only one class or use estimator.score (accuracy).
    scores = {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
    }
    return scores, pred
