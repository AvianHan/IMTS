#!/usr/bin/env python3
"""Build the reviewable Geolife approval after all src/config edits are final.

This writes a new policy file only, never commits/pushes or edits result records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from recovery_policy import ALLOWED_FILES, KIND, LEGACY_SHA256, tree_digest, validate_policy

ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "bb6136d5e93f646cad9826396fe8512800e55e17"


def build(root=ROOT):
    root = Path(root)
    changes = {}
    for name in sorted(ALLOWED_FILES):
        old = subprocess.check_output(["git", "-C", str(root), "show", f"{LEGACY_COMMIT}:{name}"])
        new = (root / name).read_bytes()
        changes[name] = {"legacy_sha256": hashlib.sha256(old).hexdigest(),
                         "corrected_sha256": hashlib.sha256(new).hexdigest(),
                         "legacy_text": old.decode("utf-8")}
    policy = {"schema_version": 1, "kind": KIND, "legacy_commit": LEGACY_COMMIT,
              "legacy_source_sha256": LEGACY_SHA256, "corrected_source_sha256": tree_digest(root),
              "changed_files": changes,
              "rationale": "Use the official non-test label vocabulary while retaining the exact inner split. "
              "Previously successful fits already contained all labels and remain numerically unchanged. "
              "Legacy GeolifeSupervised seed 0 successes are never approved. "
              "Report/CLI changes implement this explicit two-source approval only; no result is rewritten."}
    return validate_policy(policy, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(f"Policy: {args.output}\nCorrected source: {policy['corrected_source_sha256']}")
