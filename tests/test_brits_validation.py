"""Regression checks for the upstream BRITS validation-data mixup."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pypots")

from imts_benchmark.compat import BRITSHoldoutValidation, check_brits_compatibility
from imts_benchmark.models import build_model, model_settings
from imts_benchmark import compat


def test_fingerprint_survives_python_token_number_changes(monkeypatch):
    """Python 3.10/3.11 use OP=54, while Python 3.12 uses OP=55."""
    original_tokens = compat.tokenize.generate_tokens
    op = compat.tokenize.OP
    alternate_op = 10000

    def renumbered_tokens(readline):
        for token in original_tokens(readline):
            yield token._replace(type=alternate_op) if token.type == op else token

    monkeypatch.setitem(compat.tokenize.tok_name, alternate_op, "OP")
    monkeypatch.setattr(compat.tokenize, "generate_tokens", renumbered_tokens)
    check_brits_compatibility()


def test_actual_upstream_code_change_is_still_rejected(monkeypatch):
    source = compat.inspect.getsource(compat.UpstreamBRITS.fit)
    changed_source = source.replace("shuffle=True", "shuffle=False", 1)
    assert changed_source != source
    monkeypatch.setattr(compat.inspect, "getsource", lambda _: changed_source)
    with pytest.raises(RuntimeError, match="differs from the audited implementation"):
        check_brits_compatibility()


def test_fingerprint_ignores_comments_and_blank_lines(monkeypatch):
    source = compat.inspect.getsource(compat.UpstreamBRITS.fit)
    monkeypatch.setattr(compat.inspect, "getsource", lambda _: "# added comment\n\n" + source)
    check_brits_compatibility()


def _capture_fit_without_training():
    # Keep the real upstream dataset/loader construction but skip all neural
    # initialization and optimization: this tests which data reaches validation.
    model = object.__new__(BRITSHoldoutValidation)
    captured = {}
    model.batch_size = 8
    model.num_workers = 0
    model.best_model_dict = {"checkpoint": "best"}
    model.model_saving_strategy = "best"
    model.model = SimpleNamespace(
        load_state_dict=lambda state: captured.update(restored_state=state)
    )
    model._train_model = lambda train, val: captured.update(train=train, val=val)
    model._auto_save_model_if_necessary = lambda **kw: captured.update(save=kw)
    return model, captured


def test_validation_loader_contains_heldout_observations_and_labels():
    model, captured = _capture_fit_without_training()
    train = {"X": np.full((6, 4, 2), 3.0), "y": np.zeros(6, dtype=int)}
    heldout = {"X": np.full((2, 4, 2), 97.0), "y": np.ones(2, dtype=int)}

    model.fit(train, heldout)

    training_batch = next(iter(captured["train"]))
    validation_batch = next(iter(captured["val"]))
    assert len(captured["train"].dataset) == 6
    assert len(captured["val"].dataset) == 2
    np.testing.assert_array_equal(training_batch[1].numpy(), train["X"])
    np.testing.assert_array_equal(validation_batch[1].numpy(), heldout["X"])
    np.testing.assert_array_equal(validation_batch[-1].numpy(), heldout["y"])
    assert captured["restored_state"] == model.best_model_dict
    assert captured["save"] == {"confirm_saving": True}


def test_no_validation_split_still_passes_none():
    model, captured = _capture_fit_without_training()
    train = {"X": np.ones((2, 4, 1)), "y": np.array([0, 1])}
    model.fit(train)
    assert captured["val"] is None


def test_factory_records_validation_correction_and_preserves_architecture():
    check_brits_compatibility()
    model = build_model("BRITS", seed=7, smoke=True)
    settings = model_settings(model)
    assert settings["_compatibility"]["brits_heldout_validation_fix"] is True
    assert settings["model_params"]["rnn_hidden_size"] == 256
    assert settings["model_params"]["epochs"] == 1
    assert settings["model_params"]["patience"] == 1
    assert settings["random_state"] == 7
