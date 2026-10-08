"""Select utility-only checks without collecting model imports."""

from pathlib import Path

import pytest


LIGHTWEIGHT_MODULES = {
    "test_dataset_split.py",
    "test_upload_dataset_to_hf.py",
    "test_release_hygiene.py",
    "test_checkpoint_compatibility.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--lightweight",
        action="store_true",
        help="Run utility-only tests without model dependencies.",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if (
        config.getoption("--lightweight")
        and collection_path.name.startswith("test_")
        and collection_path.suffix == ".py"
        and collection_path.name not in LIGHTWEIGHT_MODULES
    ):
        return True
    return None
