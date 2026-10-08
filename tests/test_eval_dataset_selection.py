"""Exercise the dataset boundaries used by the shipped offline evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset_utils import DatasetSplitSpec, save_dataset_split
from eval_policy import build_episode_groups, resolve_dataset_selection


def _dataset(root: Path, *, splits: dict[str, str] | None = None) -> Path:
    (root / "data").mkdir(parents=True)
    episodes = root / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    (episodes / "file-000.parquet").touch()
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {"codebase_version": "v3.0", "total_episodes": 4, "splits": splits or {}}
        ),
        encoding="utf-8",
    )
    return root


def _args(data_root: Path, **overrides) -> SimpleNamespace:
    values = {
        "dataset": None,
        "dataset_repo_id": None,
        "local_data_root": data_root,
        "eval_split": "test",
        "test_ratio": None,
        "split_seed": 0,
        "split_shuffle": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _saved_split(run: Path, dataset: Path) -> Path:
    return save_dataset_split(
        run,
        DatasetSplitSpec(
            dataset_arg="demo",
            dataset_root=str(dataset),
            dataset_repo_id="local/demo",
            total_episodes=4,
            train_episode_indices=[0, 2],
            test_episode_indices=[1, 3],
            test_ratio=0.5,
            split_seed=42,
            split_shuffle=True,
        ),
    )


def test_episode_groups_preserve_view_indices_and_first_seen_order() -> None:
    dataset = SimpleNamespace(hf_dataset={"episode_index": [12, 12, 3, 12, 3]})

    assert build_episode_groups(dataset) == [(12, [0, 1, 3]), (3, [2, 4])]
    assert build_episode_groups(SimpleNamespace(hf_dataset={"episode_index": []})) == []


@pytest.mark.parametrize("split, expected", [("train", [0, 2]), ("test", [1, 3])])
def test_checkpoint_split_takes_precedence_over_dataset_metadata(
    tmp_path: Path, split: str, expected: list[int]
) -> None:
    dataset = _dataset(tmp_path / "data" / "demo", splits={"test": "0:4"})
    run = tmp_path / "run"
    split_path = _saved_split(run, dataset)

    root, repo_id, episodes, source = resolve_dataset_selection(
        args=_args(tmp_path / "data", eval_split=split),
        policy_dir=run / "checkpoints" / "last" / "pretrained_model",
    )

    assert (root, repo_id, episodes) == (dataset, "local/demo", expected)
    assert source == f"training_artifact:{split_path}"


def test_explicit_ratio_rebuilds_split_instead_of_reusing_training_artifact(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "data" / "demo")
    run = tmp_path / "run"
    _saved_split(run, dataset)

    _, _, episodes, source = resolve_dataset_selection(
        args=_args(tmp_path / "data", test_ratio=0.25),
        policy_dir=run,
    )

    assert episodes == [3]
    assert source == "cli_ratio_split"


def test_new_dataset_does_not_inherit_checkpoint_episode_selection(
    tmp_path: Path,
) -> None:
    original = _dataset(tmp_path / "data" / "original")
    target = _dataset(tmp_path / "data" / "target", splits={"val": "2:4"})
    run = tmp_path / "run"
    _saved_split(run, original)

    root, repo_id, episodes, source = resolve_dataset_selection(
        args=_args(tmp_path / "data", dataset="target", dataset_repo_id="local/target"),
        policy_dir=run,
    )

    assert (root, repo_id, episodes, source) == (
        target,
        "local/target",
        [2, 3],
        "dataset_info",
    )


def test_missing_held_out_split_requires_explicit_selection(tmp_path: Path) -> None:
    _dataset(tmp_path / "data" / "demo", splits={"train": "0:4"})

    with pytest.raises(ValueError, match="Could not resolve evaluation episodes"):
        resolve_dataset_selection(
            args=_args(tmp_path / "data", dataset="demo"),
            policy_dir=tmp_path / "unrelated_checkpoint",
        )
