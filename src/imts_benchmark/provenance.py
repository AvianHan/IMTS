"""Record source and environment; never collect credentials or arbitrary env vars."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "94644559dbe0d92bc0a10105c8d2ce300b700c97"


def verify_upstream():
    """Refuse a different pipeline, even if an environment was manually changed."""
    import pyrregular
    expected = json.loads((ROOT / "references/upstream_hashes.json").read_text())
    base = Path(pyrregular.__file__).parent
    mismatches = [path for path, digest in expected.items()
                  if not (base / path).exists() or hashlib.sha256((base / path).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f"Pyrregular source differs from pinned revision: {mismatches}")
    return {"revision": UPSTREAM_REVISION, "verified_source_files": len(expected)}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def source_digest():
    h = hashlib.sha256()
    for folder in (ROOT / "src", ROOT / "configs"):
        for path in sorted(folder.rglob("*")):
            if path.suffix in {".py", ".json"}:
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def environment():
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": dict(sorted(packages.items())), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")), "source_sha256": source_digest(),
        "expected_pyrregular_revision": UPSTREAM_REVISION,
    }


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
