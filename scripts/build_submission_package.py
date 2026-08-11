#!/usr/bin/env python3
"""Build a minimal, fully hashed submission directory from a candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from submission_utils import artifact_identity, sha256_file


REQUIRED = ("manifest.json", "model.py", "weights.pt")
OPTIONAL = ("provenance.json",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    output = args.output.resolve()
    json_out = args.json_out.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if json_out == output or json_out.is_relative_to(output):
        raise ValueError("--json-out must be outside the minimal package directory")
    missing = [name for name in REQUIRED if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(f"candidate is missing required files: {missing}")

    names = list(REQUIRED)
    names.extend(name for name in OPTIONAL if (candidate / name).is_file())
    output.mkdir(parents=True)
    for name in names:
        shutil.copy2(candidate / name, output / name)

    unexpected = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name not in names
    ]
    if unexpected:
        raise RuntimeError(f"package contains unexpected files: {unexpected}")

    receipt = {
        "status": "completed",
        "candidate": str(candidate),
        "candidate_files": artifact_identity(candidate),
        "package": str(output),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "files": artifact_identity(output),
        "file_count": len(names),
        "total_bytes": sum((output / name).stat().st_size for name in names),
        "unexpected_files": unexpected,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
