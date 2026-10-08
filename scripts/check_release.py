#!/usr/bin/env python3
"""Check source-release hygiene without importing the model stack.

This is a focused regression guard, not a comprehensive secret-scanning service.
Only paths and line numbers are reported for potential credentials.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "outputs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "build",
    "dist",
    "wandb",
}
BLOCKED_SUFFIXES = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bag",
    ".mcap",
    ".parquet",
    ".so",
}
CREDENTIALS = re.compile(
    rb"hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9_]{20,}|"
    rb"gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
LOCAL_PATH = re.compile(rb"/(?:home|Users)/[A-Za-z0-9_.-]+/")
REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "configs/trace_act.yaml",
    "configs/trace_diffusion.yaml",
    "examples/smoke_trace.py",
    "docs/installation.md",
    "docs/training.md",
    "docs/evaluation.md",
    "docs/architecture.md",
    "docs/release-scope.md",
    "docs/maintainers/validation.md",
    "docs/i18n/README.zh-CN.md",
    "docs/i18n/README.ja.md",
    "docs/i18n/README.ko.md",
)


def source_files(root: Path) -> list[Path]:
    """Return an ordered source manifest for validation and packaging."""
    result = []
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if any(p in EXCLUDED_DIRS or p.endswith(".egg-info") for p in parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Source release cannot contain symlinks: {path.relative_to(root)}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            result.append(path)
    return sorted(result)


def check_release(root: Path) -> list[str]:
    """Return actionable failures without printing file contents."""
    errors = [f"Missing required file: {name}" for name in REQUIRED if not (root / name).is_file()]
    try:
        files = source_files(root)
    except ValueError as exc:
        return errors + [str(exc)]
    for path in files:
        relative = path.relative_to(root)
        if path.suffix in BLOCKED_SUFFIXES or path.name.startswith(".env"):
            errors.append(f"Non-source or credential file: {relative}")
        if any(word in relative.name.lower() for word in ("ablation", "sweep")):
            errors.append(f"Experiment-only file in main-method release: {relative}")
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        for line_no, line in enumerate(data.splitlines(), 1):
            if CREDENTIALS.search(line):
                errors.append(f"Potential credential: {relative}:{line_no}")
            if relative.parts[0] != "depends" and LOCAL_PATH.search(line):
                errors.append(f"Machine-specific path: {relative}:{line_no}")
        if path.suffix == ".md" and relative.parts[0] != "depends":
            text = data.decode("utf-8")
            for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", text):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                destination = (path.parent / target.split("#", 1)[0]).resolve()
                if not destination.is_relative_to(root.resolve()) or not destination.exists():
                    errors.append(f"Broken local link: {relative} -> {target}")
    main_readme = root / "README.md"
    if main_readme.is_file():
        blocks = re.findall(r"```(?:bash|bibtex)\n(.*?)```", main_readme.read_text(), re.S)
        for translation in (root / "docs/i18n").glob("README.*.md"):
            translated = re.findall(r"```(?:bash|bibtex)\n(.*?)```", translation.read_text(), re.S)
            if translated != blocks:
                errors.append(
                    f"Commands/citation differ from English: {translation.relative_to(root)}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    failures = check_release(args.root.resolve())
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print(
        f"Release checks passed: {len(source_files(args.root))} source files; links and README commands aligned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
