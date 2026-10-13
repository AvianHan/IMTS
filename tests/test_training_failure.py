import io
import numpy as np
import pytest


def test_pypots_error_after_checkpoint_is_not_success():
    pytest.importorskip("pypots")
    import torch
    from pypots.classification.timesnet import TimesNet
    from imts_benchmark.training import checked_training

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = TimesNet(n_steps=8, n_features=2, n_classes=2, n_layers=1,
                    d_model=4, d_ffn=8, n_kernels=1, top_k=1,
                    epochs=2, batch_size=8, patience=2, device="cpu", verbose=False)
    rng = np.random.default_rng(2)
    train = {"X": rng.normal(size=(8, 8, 2)), "y": np.arange(8) % 2}
    val = {"X": rng.normal(size=(4, 8, 2)), "y": np.arange(4) % 2}
    training_calls = []

    def fail_second_epoch(module, inputs):
        if module.training:
            training_calls.append(True)
            if len(training_calls) == 2:
                raise RuntimeError("injected epoch-2 failure")

    hook = model.model.register_forward_pre_hook(fail_second_epoch)
    try:
        with pytest.raises(RuntimeError, match="aborted despite returning"):
            with checked_training(True, io.StringIO()):
                model.fit(train, val)
        assert model.best_model_dict is not None
        assert len(training_calls) == 2
    finally:
        hook.remove()
