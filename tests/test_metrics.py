import numpy as np
import pytest
from imts_benchmark.metrics import evaluate


def test_macro_f1_not_accuracy_and_class_column_order():
    # Imbalanced example: predicting only majority class yields .9 accuracy,
    # but macro-F1 is about .474. Probability columns use estimator.classes_.
    y = np.array(["common"] * 9 + ["rare"])
    p = np.tile([0.2, 0.8], (10, 1))
    scores, pred = evaluate(y, p, np.array(["rare", "common"]))
    assert scores["accuracy"] == .9
    assert scores["macro_f1"] == pytest.approx(9 / 19)
    assert np.all(pred == "common")


@pytest.mark.parametrize("p", [np.array([[float("nan"), 0]]), np.array([[1., 1.]]), np.array([[-.1, 1.1]])])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        evaluate(np.array([0]), p, np.array([0, 1]))


def test_unknown_test_label_is_not_silently_dropped():
    with pytest.raises(ValueError, match="absent"):
        evaluate(np.array([2]), np.array([[.2, .8]]), np.array([0, 1]))
