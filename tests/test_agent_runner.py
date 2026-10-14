"""Audit data boundaries, persistence, deterministic fitting, and test-once gates."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from imts_agents import runner
from imts_agents.__main__ import main


class TinyClassifier(nn.Module):
    training_calls = 0
    test_calls = 0

    def __init__(self, n_features, n_channels, n_classes, variant, **kwargs):
        super().__init__()
        self.variant = variant
        self.head = nn.Linear(n_features, n_classes)
        self.policy = nn.Parameter(torch.zeros(()))

    def forward(self, X, mask, *, return_actions=False):
        if self.training:
            type(self).training_calls += 1
        if X[..., -1].max().item() > 90:
            type(self).test_calls += 1
        features = X.mean((1, 2))
        logits = self.head(features)
        batch = len(X)
        distribution = torch.distributions.Bernoulli(logits=self.policy.expand(batch))
        actions = distribution.sample() if self.training else distribution.probs.ge(0.5).float()
        count = mask.sum((1, 2)).float()[:, None].expand(-1, len(self.variant.role_scopes))
        result = {"logits": logits, "policy_log_prob": distribution.log_prob(actions),
                  "policy_entropy": distribution.entropy(), "updates": 1 + actions,
                  "action_counts": torch.ones(batch, 4, device=X.device),
                  "input_gate_sum": count, "input_gate_count": count, "input_gate_squared_sum": count}
        if return_actions:
            result["actions"] = torch.where(mask.any(-1, keepdim=True), 0, -1)
        return result


@pytest.fixture
def fake_data(monkeypatch):
    rng = np.random.default_rng(93)
    fields = {}
    ids = {}
    for split, n in (("train", 16), ("val", 6), ("test", 6)):
        y = np.arange(n) % 2
        X = rng.normal(size=(n, 2, 2, 3)).astype(np.float32)
        X[..., 0] += y[:, None, None] * 2
        X[..., -1] = 100 if split == "test" else 0
        fields.update({f"X_{split}": X, f"mask_{split}": np.ones(X.shape[:-1], bool), f"y_{split}": y})
        ids[split] = [f"{split}-{i}" for i in range(n)]
    data = SimpleNamespace(**fields, classes=np.array(["a", "b"]),
                           metadata={"sample_ids": ids, "sample_ids_sha256": {k: runner._digest(v) for k, v in ids.items()}})
    import imts_agents.model
    TinyClassifier.training_calls = TinyClassifier.test_calls = 0
    monkeypatch.setattr(imts_agents.model, "AgentClassifier", TinyClassifier)
    monkeypatch.setattr(runner, "prepare_dataset", lambda *args, **kwargs: deepcopy(data))
    monkeypatch.setattr(runner, "environment", lambda: {"source_sha256": "source-a", "packages": {"torch": torch.__version__}})
    return data


def run_small(tmp_path, **kwargs):
    options = {"output_dir": tmp_path, "epochs": 2, "patience": 1, "batch_size": 8, "hidden": 4,
               "n_patches": 2, "threads": 1}
    options.update(kwargs)
    return runner.execute(**options)


def test_validation_default_never_evaluates_test_and_reconstructs_metrics(fake_data, tmp_path):
    result, folder = run_small(tmp_path)
    assert result["status"] == "success"
    assert result["profile"] == "validation"
    assert result["test_evaluated"] is False
    assert TinyClassifier.test_calls == 0
    assert result["memory"]["cpu_peak_rss_bytes"] is None
    assert (folder / "best.pt").is_file()
    assert not (folder / "test_predictions.npz").exists()
    assert not (folder / "result.json.tmp").exists()
    persisted = json.loads((folder / "result.json").read_text())
    assert persisted["metrics"] == result["metrics"]
    with np.load(folder / "val_predictions.npz", allow_pickle=False) as saved:
        metrics, _ = runner.evaluate(saved["y_true"], saved["probabilities"], saved["classes"])
        assert metrics == result["metrics"]
    assert 1 <= result["best_epoch"] <= result["epochs_completed"] <= 2
    diagnostics = json.loads((folder / result["diagnostics_file"]).read_text())
    assert result["diagnostics_sha256"] == hashlib.sha256((folder / "diagnostics.json").read_bytes()).hexdigest()
    assert diagnostics["split"] == "validation"
    assert diagnostics["checkpoint_epoch"] == result["best_epoch"]
    assert diagnostics["history_access"] == "prefix"
    assert diagnostics["input_write"] == "broadcast"
    assert diagnostics["input_write_gate"]["mean"] == [1, 1, 1]
    assert diagnostics["input_write_gate"]["population_std"] == [0, 0, 0]
    assert diagnostics["input_write_gate"]["valid_token_count"] == [24, 24, 24]
    assert diagnostics["action_paths"]["nonempty_patches"] == 12
    assert diagnostics["action_paths"]["distribution"] == [{"actions": ["T"], "count": 12, "fraction": 1.0}]
    assert all(result["validation_checks"].values())


def test_same_seed_reproduces_predictions(fake_data, tmp_path):
    first, first_dir = run_small(tmp_path)
    second, second_dir = run_small(tmp_path)
    assert first["comparison_protocol_sha256"] == second["comparison_protocol_sha256"]
    with np.load(first_dir / "val_predictions.npz") as one, np.load(second_dir / "val_predictions.npz") as two:
        np.testing.assert_array_equal(one["probabilities"], two["probabilities"])


def test_failure_persists_error_without_scores(fake_data, monkeypatch, tmp_path):
    broken = deepcopy(fake_data)
    broken.X_train[0, 0, 0, 0] = np.nan
    monkeypatch.setattr(runner, "prepare_dataset", lambda *args, **kwargs: broken)
    with pytest.raises(ValueError, match="Non-finite"):
        run_small(tmp_path)
    files = list(tmp_path.rglob("result.json"))
    assert len(files) == 1
    result = json.loads(files[0].read_text())
    assert result["status"] == "failed"
    assert "metrics" not in result
    assert (files[0].parent / "error.txt").exists()
    assert runner.summarize(tmp_path)["incomplete"][0]["metrics"] is None


def test_smoke_is_one_epoch_and_cannot_evaluate_test(fake_data, tmp_path):
    result, folder = run_small(tmp_path, smoke=True, epochs=100)
    assert result["profile"] == "smoke"
    assert result["epochs_completed"] == 1
    assert result["stop_reason"] == "smoke_epoch_limit"
    with pytest.raises(ValueError, match="complete validation"):
        runner.evaluate_checkpoint(folder)
    with pytest.raises(ValueError, match="Smoke diagnostics"):
        run_small(tmp_path, smoke=True, evaluate_test=True)


def test_dynamic_policy_has_a_finite_training_objective(fake_data, tmp_path):
    result, folder = run_small(tmp_path, variant="dynamic_feedback")
    assert result["status"] == "success"
    history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
    assert any(row["train"]["policy_entropy"] > 0 for row in history)
    assert all(np.isfinite(row["train"]["objective"]) for row in history)
    state = torch.load(folder / "best.pt", weights_only=True)["model_state_dict"]
    assert abs(state["policy"].item()) > 0


def test_checkpoint_evaluation_does_not_retrain_and_is_test_once(fake_data, tmp_path):
    _, folder = run_small(tmp_path)
    training_calls = TinyClassifier.training_calls
    result, _ = runner.evaluate_checkpoint(folder)
    assert TinyClassifier.training_calls == training_calls
    assert TinyClassifier.test_calls > 0
    assert result["profile"] == "test"
    assert result["test_evaluation_status"] == "success"
    assert (folder / "test_predictions.npz").exists()
    with pytest.raises(ValueError):
        runner.evaluate_checkpoint(folder)


def test_checkpoint_source_change_is_blocked_before_test(fake_data, monkeypatch, tmp_path):
    _, folder = run_small(tmp_path)
    monkeypatch.setattr(runner, "environment", lambda: {"source_sha256": "changed"})
    with pytest.raises(ValueError, match="Source differs"):
        runner.evaluate_checkpoint(folder)
    assert TinyClassifier.test_calls == 0


def test_checkpoint_data_change_is_blocked_before_test(fake_data, monkeypatch, tmp_path):
    _, folder = run_small(tmp_path)
    changed = deepcopy(fake_data)
    changed.X_test[0, 0, 0, 0] += 1
    monkeypatch.setattr(runner, "prepare_dataset", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="Prepared data"):
        runner.evaluate_checkpoint(folder)
    assert TinyClassifier.test_calls == 0


def test_summary_keeps_protocols_and_repeated_attempts_separate(fake_data, tmp_path):
    run_small(tmp_path, variant="core_none")
    run_small(tmp_path, variant="core_both")
    run_small(tmp_path, variant="core_both")
    run_small(tmp_path, variant="core_both", seed=1)
    run_small(tmp_path, variant="core_both", smoke=True)
    summary = runner.summarize(tmp_path)
    assert len(summary["protocol_groups"]) == 3
    assert sorted(len(group["runs"]) for group in summary["protocol_groups"].values()) == [1, 1, 3]


def test_cli_reports_failure_with_nonzero_exit(monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError("deliberate failure")
    monkeypatch.setattr(runner, "execute", fail)
    assert main(["run", "--variant", "core_both"]) == 1
    assert "deliberate failure" in capsys.readouterr().err


@pytest.mark.parametrize("options", [{"seed": -1}, {"epochs": 0}, {"learning_rate": float("nan")},
                                     {"learning_rate": 0}, {"device": "gpu"}, {"dataset": "synthetic"}])
def test_config_invalid_values_are_rejected(options):
    with pytest.raises(ValueError):
        runner.RunConfig(**options).validate()


class DiagnosticClassifier(nn.Module):
    """Deliberately sparse diagnostics test weighted reductions and trace padding."""

    def __init__(self, paths=None):
        super().__init__()
        self.variant = SimpleNamespace(role_scopes=("T", "S", "ST"))
        self.paths = paths

    def forward(self, X, mask, *, return_actions=False):
        gate = torch.stack((X[..., 0], X[..., 0] * 0 + 0.5, X[..., 0] * 0 + 1), -1)
        gate = torch.where(mask[..., None], gate, 0)
        count = mask.sum((1, 2)).float()[:, None].expand(-1, 3)
        result = {"logits": X.new_zeros(len(X), 2), "updates": X.new_ones(len(X)),
                  "action_counts": X.new_ones(len(X), 4),
                  "input_gate_sum": gate.sum((1, 2)), "input_gate_count": count,
                  "input_gate_squared_sum": gate.square().sum((1, 2))}
        if return_actions:
            result["actions"] = self.paths
        return result


def test_gate_statistics_weight_valid_tokens_and_exclude_masked_values():
    X = torch.tensor([[[[0.1], [9.0]], [[9.0], [9.0]]],
                      [[[0.2], [0.3]], [[0.4], [9.0]]],
                      [[[0.9], [9.0]], [[9.0], [9.0]]]])
    mask = X[..., 0] != 9
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, mask, torch.tensor([0, 1, 0])), batch_size=2)
    predicted = runner._predict(DiagnosticClassifier(), loader, "cpu", 2)
    stats = predicted[4]["input_write_gate"]
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    assert len(predicted) == 5
    assert stats["valid_token_count"] == [5, 5, 5]
    np.testing.assert_allclose(stats["mean"], [values.mean(), 0.5, 1], atol=1e-7)
    np.testing.assert_allclose(stats["population_std"], [values.std(), 0, 0], atol=1e-7)
    assert "not FLOPs" in predicted[4]["cost_scope"]


def test_gate_statistics_with_no_valid_tokens_is_json_safe():
    stats = runner._gate_statistics(np.zeros(3), np.zeros(3), np.zeros(3), ["T", "S", "ST"])
    assert stats["mean"] == [None, None, None]
    assert stats["population_std"] == [None, None, None]
    json.dumps(stats, allow_nan=False)


def test_action_paths_count_valid_patches_and_commit_without_full_trace():
    X = torch.zeros(2, 3, 1, 1)
    mask = torch.tensor([[[True], [False], [True]], [[True], [True], [False]]])
    paths = torch.tensor([[[0, 3, -1], [-1, -1, -1], [0, 1, 2]],
                          [[0, 3, -1], [2, 1, 0], [-1, -1, -1]]])
    loader = [(X, mask, torch.tensor([0, 1]))]
    diagnostics = runner._action_path_diagnostics(DiagnosticClassifier(paths), loader, "cpu")
    assert diagnostics["nonempty_patches"] == 4
    assert diagnostics["empty_patches"] == 2
    assert diagnostics["unique_paths"] == 3
    assert diagnostics["commit_patch_fraction"] == 0.5
    assert diagnostics["distribution"][0] == {"actions": ["T", "COMMIT"], "count": 2, "fraction": 0.5}
    assert sum(row["count"] for row in diagnostics["distribution"]) == 4


@pytest.mark.parametrize("path", [[0, 3, 1], [0, -1, 2], [0, 4, -1]])
def test_action_path_corruption_is_rejected(path):
    loader = [(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool), torch.zeros(1, dtype=torch.long))]
    with pytest.raises(ValueError):
        runner._action_path_diagnostics(DiagnosticClassifier(torch.tensor([[path]])), loader, "cpu")
