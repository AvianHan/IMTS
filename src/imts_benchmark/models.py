"""Seeded copies of Pyrregular's six selected paper baselines.

The implementations, preprocessing assumptions, and architecture defaults come
from the pinned upstream dependency.  In particular, the paper's ROCKET label
means MiniRocketMultivariateVariable + LightGBM, and BORF uses aeon's BORF.
Heavy model libraries are imported only when their baseline is requested.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any


MODEL_NAMES = ("ROCKET", "BORF", "LGBM", "RIFC", "TIMESNET", "BRITS")
DEEP_MODELS = frozenset({"TIMESNET", "BRITS"})
_SOURCES = {
    "ROCKET": "pyrregular.models.rocket:rocket_pipeline",
    "BORF": "pyrregular.models.aeon_borf:aeon_borf_pipeline",
    "LGBM": "pyrregular.models.lgbm:lgbm_pipeline",
    "RIFC": "pyrregular.models.rifc:rifc_pipeline",
    "TIMESNET": "pyrregular.models.timesnet:timesnet_pipeline",
    "BRITS": "pyrregular.models.brits:brits_pipeline",
}


def normalize_name(name: str) -> str:
    """Return the canonical paper name, accepting LightGBM as an alias."""
    normalized = name.strip().upper()
    if normalized == "LIGHTGBM":
        normalized = "LGBM"
    if normalized not in MODEL_NAMES:
        raise ValueError(
            f"Unknown baseline {name!r}; choose from {', '.join(MODEL_NAMES)}"
        )
    return normalized


def model_source(name: str) -> str:
    """Return the exact upstream module and pipeline used for this baseline."""
    return _SOURCES[normalize_name(name)]


def _seed_process(seed: int, *, deep: bool) -> None:
    """Seed RNGs used by a run without loading torch for classical models."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if deep:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_model(
    name: str,
    seed: int,
    smoke: bool = False,
    device: str = "cpu",
) -> Any:
    """Return a fresh, unfitted official pipeline with explicit run seeds.

    Full runs preserve upstream model defaults apart from RNG seeds and the
    requested device. BRITS applies the documented PyPOTS 1.5 validation-loader
    correction in ``compat.py``. Smoke runs change only deep-model ``epochs`` and
    ``patience`` to 1; they retain the actual architecture.  They are pipeline
    checks, not benchmark results.  Classical model settings do not change.

    This seeds Python/NumPy and, for neural baselines, torch.  It does not promise
    bitwise equality across software versions, devices, or nondeterministic GPU
    kernels.  The caller controls CPU thread limits and records its environment.
    """
    name = normalize_name(name)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    deep = name in DEEP_MODELS
    if deep and device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; run on an allocated GPU node or explicitly choose --device cpu.")
    _seed_process(seed, deep=deep)

    from pyrregular.models import get_paper_model

    # get_paper_model clones the upstream singleton, so neither these parameter
    # changes nor fit() can contaminate another run's model.
    model = get_paper_model(name)
    if name == "BRITS":
        from imts_benchmark.compat import BRITSHoldoutValidation, check_brits_compatibility

        check_brits_compatibility()
        model.set_params(model=BRITSHoldoutValidation)
    parameters = model.get_params(deep=True)
    seed_parameters = {
        key: seed
        for key in parameters
        if key == "random_state" or key.endswith("__random_state")
    }
    if seed_parameters:
        model.set_params(**seed_parameters)

    if deep:
        # model_params is a dictionary rather than a nested sklearn estimator;
        # copy it instead of mutating the upstream configuration in place.
        model_parameters = dict(model.get_params(deep=False)["model_params"])
        model_parameters["device"] = device
        if smoke:
            model_parameters.update(epochs=1, patience=1)
        model.set_params(model_params=model_parameters)
    return model


def _qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", value.__class__.__module__)
    name = getattr(value, "__qualname__", None)
    if name is None:
        name = getattr(value, "__name__", value.__class__.__qualname__)
    return f"{module}.{name}"


def _json_safe(value: Any) -> Any:
    """Represent effective parameters without unstable object addresses."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, type):
        return _qualified_name(value)
    if hasattr(value, "get_params"):
        return {
            "class": _qualified_name(value.__class__),
            "parameters": _json_safe(value.get_params(deep=False)),
        }
    if callable(value):
        return _qualified_name(value)
    # NumPy arrays and scalar types, without importing NumPy at module import.
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def model_settings(model: Any) -> dict[str, Any]:
    """Return JSON-safe exposed settings for an audit record.

    This records constructor parameters exposed by ``get_params``, not learned
    weights.  Defaults hidden inside third-party implementations are identified
    by the recorded dependency versions and pinned source revision.
    """
    settings = _json_safe(model.get_params(deep=True))
    compatibility = getattr(getattr(model, "model", None), "imts_compatibility", None)
    if compatibility is not None:
        settings["_compatibility"] = _json_safe(compatibility)
    return settings
