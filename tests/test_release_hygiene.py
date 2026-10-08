"""Regression tests for source packaging and translation consistency."""

from pathlib import Path

import pytest

from check_release import REQUIRED, check_release, source_files


def make_release(root: Path) -> None:
    for name in REQUIRED:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("")


def test_credentials_report_only_location(tmp_path: Path) -> None:
    make_release(tmp_path)
    secret = "hf_" + "a" * 30
    (tmp_path / "example.py").write_text(f'TOKEN = "{secret}"\n')
    errors = check_release(tmp_path)
    assert "Potential credential: example.py:1" in errors
    assert all(secret not in error for error in errors)


def test_translated_commands_cannot_drift(tmp_path: Path) -> None:
    make_release(tmp_path)
    (tmp_path / "README.md").write_text("```bash\npython example.py\n```\n")
    assert any("Commands/citation differ" in e for e in check_release(tmp_path))


def test_archive_manifest_excludes_local_artifacts(tmp_path: Path) -> None:
    for name in (
        "src/model.py",
        "outputs/model.pt",
        ".git/config",
        ".venv/secret.py",
        "src/__pycache__/x.pyc",
    ):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("test")
    assert [p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)] == ["src/model.py"]


def test_symlink_cannot_escape_source_tree(tmp_path: Path) -> None:
    (tmp_path / "outside.py").symlink_to(tmp_path.parent / "not-source.py")
    with pytest.raises(ValueError, match="symlinks"):
        source_files(tmp_path)
