"""TRACE configurations round-trip through the public LeRobot checkpoint API."""

import dataclasses
import json
from pathlib import Path

import pytest

from policy_imports import ensure_lerobot_policy_imports

ensure_lerobot_policy_imports("prism_diffusion")

from lerobot_policy_prism_diffusion.configuration_diffusion import PrismDiffusionConfig
from lerobot_policy_streaming_act.configuration_streaming_act import StreamingACTConfig


@pytest.mark.parametrize("config_class", [StreamingACTConfig, PrismDiffusionConfig])
def test_config_round_trip_has_no_architecture_switches(config_class, tmp_path: Path) -> None:
    config = config_class(device="cpu", history_length=4, signature_dim=6)
    config.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))

    assert "use_path_signature" not in payload
    assert "use_visual_prefix_memory" not in payload
    assert "use_first_frame_anchor" not in payload
    assert "slot_memory_balance_loss_coef" not in payload
    restored = config_class.from_pretrained(tmp_path)

    assert dataclasses.asdict(restored) == dataclasses.asdict(config)
    assert restored.use_path_signature is True
    with pytest.raises(AttributeError):
        restored.use_path_signature = False


@pytest.mark.parametrize("config_class", [StreamingACTConfig, PrismDiffusionConfig])
def test_full_historical_checkpoint_loads_and_invalid_one_fails(
    config_class, tmp_path: Path
) -> None:
    config = config_class(device="cpu", history_length=4, signature_dim=6)
    config.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.update(use_path_signature=True, use_delta_signature=True, use_visual_prefix_memory=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert config_class.from_pretrained(tmp_path).signature_dim == 6

    payload["use_path_signature"] = False
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="use_path_signature"):
        config_class.from_pretrained(tmp_path)
