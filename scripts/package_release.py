#!/usr/bin/env python3
"""Build a reviewed source archive without Git history, data or checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from check_release import ROOT, check_release, source_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/releases/trace-source.zip")
    args = parser.parse_args()
    errors = check_release(ROOT)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    output = args.output.expanduser().resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "outputs"):
        parser.error("Use an output under outputs/ or outside the source tree.")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        parser.error("Output already exists; choose a new archive name.")
    files = source_files(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(f"trace/{relative}", date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.stat().st_mode & 0o111 else 0o100644) << 16
            archive.writestr(info, data)
            manifest.append(
                {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".manifest.json").write_text(
        json.dumps({"archive": output.name, "sha256": digest, "files": manifest}, indent=2) + "\n"
    )
    print(f"Created {output.name}: {len(files)} files; SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
