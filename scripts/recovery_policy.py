"""Explicit, narrowly scoped provenance approval for the Geolife vocabulary fix.

Policies never rewrite result records. Reconstructing the legacy tree proves
that every hashed source/config byte outside the listed changes is unchanged.
This helper is frozen alongside each submitted worker.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LEGACY_SHA256 = "4ce2704ecb1037b5031abe9ce099e93269e2a10d9bc591981594fb091858f8e0"
KIND = "geolife-label-vocabulary-v1"
ALLOWED_FILES = {"src/imts_agents/data.py", "src/imts_agents/report.py",
                 "src/imts_agents/__main__.py"}


def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def tree_digest(root, overrides=None):
    root, overrides = Path(root), overrides or {}
    digest = hashlib.sha256()
    for directory in (root / "src", root / "configs"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"}:
                name = str(path.relative_to(root))
                digest.update(name.encode())
                digest.update(overrides[name] if name in overrides else path.read_bytes())
    return digest.hexdigest()


def validate_policy(policy, root):
    if (not isinstance(policy, dict) or policy.get("schema_version") != 1
            or policy.get("kind") != KIND or policy.get("legacy_source_sha256") != LEGACY_SHA256):
        raise ValueError("Unrecognized recovery policy; only the audited Geolife fix is allowed")
    corrected = policy.get("corrected_source_sha256")
    if corrected == LEGACY_SHA256 or corrected != tree_digest(root):
        raise ValueError("Recovery policy does not match the current corrected src/configs")
    changes = policy.get("changed_files")
    if not isinstance(changes, dict) or set(changes) != ALLOWED_FILES:
        raise ValueError("Recovery policy changed-file scope is not the approved Geolife fix")
    old = {}
    for name, item in changes.items():
        if not isinstance(item, dict) or not isinstance(item.get("legacy_text"), str):
            raise ValueError("Recovery policy is missing the exact legacy source bytes")
        old[name] = item["legacy_text"].encode()
        if hashlib.sha256(old[name]).hexdigest() != item.get("legacy_sha256"):
            raise ValueError(f"Recovery legacy file hash mismatch: {name}")
        if hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() != item.get("corrected_sha256"):
            raise ValueError(f"Recovery corrected file hash mismatch: {name}")
    if tree_digest(root, old) != LEGACY_SHA256:
        raise ValueError("Recovery scope exceeded: reconstructed legacy source/config tree differs")
    return policy


def load_policy(path, root):
    return validate_policy(json.loads(Path(path).read_text()), root)


def source_allowed(policy, source, dataset, seed):
    if source == policy["corrected_source_sha256"]:
        return True
    return (source == LEGACY_SHA256
            and not (dataset == "GeolifeSupervised" and seed == 0))


def summary(policy):
    return {"kind": policy["kind"], "policy_sha256": policy_digest(policy),
            "legacy_source_sha256": LEGACY_SHA256,
            "corrected_source_sha256": policy["corrected_source_sha256"],
            "excluded_legacy_success": {"dataset": "GeolifeSupervised", "seed": 0}}
