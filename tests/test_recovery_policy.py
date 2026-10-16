"""Integrity checks for one explicit, audited recovery policy."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_policy_test", ROOT / "scripts/recovery_policy.py")
policy_helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy_helper)


def test_policy_reconstructs_exact_legacy_source():
    policy = policy_helper.load_policy(ROOT / "docs/geolife-recovery-policy.json", ROOT)
    assert policy["legacy_source_sha256"] == policy_helper.LEGACY_SHA256
    assert policy["corrected_source_sha256"] == policy_helper.tree_digest(ROOT)
    assert len(policy_helper.policy_digest(policy)) == 64


def test_unrelated_source_change_is_not_approved_even_if_current_hash_is_updated(tmp_path):
    policy = json.loads((ROOT / "docs/geolife-recovery-policy.json").read_text())
    for name in ("src", "configs"):
        shutil.copytree(ROOT / name, tmp_path / name)
    (tmp_path / "src/unapproved.py").write_text("changed = True\n")
    policy["corrected_source_sha256"] = policy_helper.tree_digest(tmp_path)
    with pytest.raises(ValueError, match="scope exceeded"):
        policy_helper.validate_policy(policy, tmp_path)


def test_modified_policy_scope_and_old_bytes_are_rejected():
    policy = json.loads((ROOT / "docs/geolife-recovery-policy.json").read_text())
    policy["changed_files"]["src/imts_agents/data.py"]["legacy_text"] += "# tampered\n"
    with pytest.raises(ValueError, match="legacy file hash"):
        policy_helper.validate_policy(policy, ROOT)
