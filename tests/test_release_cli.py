"""Protect the full-method release boundary at config and command-line entry points."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval_helpers import load_pretrained_config_from_pretrained_dir, resolve_signature_backend
from eval_policy import parse_args as parse_eval_args
from train_policy import parse_args as parse_train_args

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "policy, recipe",
    [("streaming_act", "trace_act.yaml"), ("prism_diffusion", "trace_diffusion.yaml")],
)
def test_reference_recipe_selects_full_trace(policy: str, recipe: str) -> None:
    args = parse_train_args(
        [
            "--policy",
            policy,
            "--defaults-path",
            str(ROOT / "configs" / recipe),
            "--dataset",
            "example/local",
        ]
    )
    assert args.policy == policy
    assert args.use_path_signature and args.use_delta_signature
    assert args.use_visual_prefix_memory and args.use_signature_indexed_slot_memory
    assert args.slot_memory_num_slots == 4
    assert (args.prefix_train_max_steps, args.prefix_frame_stride) == (32, 4)
    assert args.signature_backend == "signatory"
    assert args.slot_memory_balance_loss_coef == 0
    assert args.slot_memory_identity_scale == 0.1

    evaluated = parse_eval_args(
        ["--policy", policy, "--policy-path", "example/checkpoint", "--dataset", "example/local"]
    )
    assert evaluated.signature_backend == "signatory"


@pytest.mark.parametrize("policy", ["streaming_act", "prism_diffusion"])
def test_capacity_parameters_remain_configurable(policy: str, tmp_path: Path) -> None:
    recipe = tmp_path / "capacity.yaml"
    recipe.write_text(
        "train:\n  slot_memory_num_slots: 6\n  prefix_train_max_steps: 48\n",
        encoding="utf-8",
    )
    args = parse_train_args(
        [
            "--policy",
            policy,
            "--dataset",
            "example/local",
            "--defaults-path",
            str(recipe),
            "--signature-depth",
            "2",
            "--prefix-frame-stride",
            "3",
        ]
    )
    assert args.slot_memory_num_slots == 6
    assert (args.prefix_train_max_steps, args.prefix_frame_stride) == (48, 3)
    assert args.signature_depth == 2
    assert args.use_path_signature and args.use_visual_prefix_memory


@pytest.mark.parametrize("policy", ["act", "diffusion", "smolvla", "pi05", "xvla"])
@pytest.mark.parametrize("parse", [parse_train_args, parse_eval_args])
def test_other_policy_entry_points_are_unavailable(parse, policy: str) -> None:
    with pytest.raises(SystemExit) as error:
        parse(["--dataset", "example/local", "--policy", policy])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "option",
    [
        "--disable-path-signature",
        "--disable-delta-signature",
        "--disable-visual-prefix-memory",
        "--disable-signature-indexed-slot-memory",
        "--enable-signature-conditioned-visual-prefix-memory",
        "--disable-memory-conditioned-encoder-film",
    ],
)
def test_component_removal_flags_are_unavailable(option: str) -> None:
    with pytest.raises(SystemExit) as error:
        parse_train_args(["--dataset", "example/local", option])
    assert error.value.code == 2


def test_yaml_cannot_override_full_method_components(tmp_path: Path) -> None:
    recipe = tmp_path / "changed.yaml"
    recipe.write_text("train:\n  use_path_signature: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Full TRACE requires use_path_signature"):
        parse_train_args(["--dataset", "example/local", "--defaults-path", str(recipe)])


def test_incompatible_checkpoint_is_rejected_before_unknown_fields_are_dropped(
    tmp_path: Path,
) -> None:
    @dataclass
    class Config:
        name: str = "test"

    (tmp_path / "config.json").write_text(
        json.dumps({"name": "test", "use_path_signature": False, "unused": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_pretrained_config_from_pretrained_dir(Config, tmp_path)


def test_signature_backend_has_no_fallback() -> None:
    with pytest.raises(ValueError, match="requires the signatory"):
        resolve_signature_backend("simple")
