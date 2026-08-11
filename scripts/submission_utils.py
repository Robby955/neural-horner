"""Utilities for loading the entry class selected by a submission manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_submission(submission: str | Path):
    """Load and initialize the exact entry class declared in manifest.json."""
    submission_dir = Path(submission).resolve()
    manifest_path = submission_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry_class = manifest["entry_class"]
    module_name, class_name = entry_class.rsplit(".", 1)
    module_path = submission_dir / (module_name.replace(".", "/") + ".py")
    if not module_path.is_file():
        raise FileNotFoundError(
            f"manifest entry module {module_name!r} not found at {module_path}"
        )

    identity = hashlib.sha256(
        f"{submission_dir}:{entry_class}".encode()
    ).hexdigest()[:16]
    import_name = f"submission_{identity}_{module_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(import_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load submission module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[import_name] = module
    sys.path.insert(0, str(submission_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    entry_type = getattr(module, class_name)
    model = entry_type()
    model.load(str(submission_dir))
    return submission_dir, manifest, module, model


def artifact_identity(submission: str | Path) -> dict[str, str]:
    submission_dir = Path(submission).resolve()
    names = ["manifest.json", "model.py", "weights.pt"]
    if (submission_dir / "provenance.json").is_file():
        names.append("provenance.json")
    return {
        name: sha256_file(submission_dir / name)
        for name in names
    }
