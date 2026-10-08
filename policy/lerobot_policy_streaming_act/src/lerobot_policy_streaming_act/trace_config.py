"""Fixed TRACE architecture and compatibility with full-method checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


_TRUE_FIELDS = (
    "use_path_signature",
    "use_delta_signature",
    "use_prefix_sequence_training",
    "use_visual_prefix_memory",
    "use_signature_indexed_slot_memory",
    "use_memory_conditioned_encoder_film",
    "slot_memory_use_delta_routing",
    "slot_memory_use_softmax_routing",
    "slot_memory_use_readout_pooling",
)
_FALSE_FIELDS = (
    "use_signature_conditioned_visual_prefix_memory",
    "use_first_frame_anchor",
    "use_first_frame_anchor_in_slot_routing",
)
_ZERO_FIELDS = (
    "slot_memory_balance_loss_coef",
    "slot_memory_entropy_loss_coef",
    "slot_memory_consistency_loss_coef",
)


def normalize_trace_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and remove historical fixed fields before parsing a checkpoint.

    New configurations omit these fields. Existing full-method checkpoints retain
    their parameter names and can be loaded; incompatible architectures fail
    before loading weights. Unknown fields remain available to the caller's
    configuration parser.
    """
    result = dict(payload)
    for names, expected in ((_TRUE_FIELDS, True), (_FALSE_FIELDS, False)):
        for name in names:
            if name in result and result.pop(name) is not expected:
                raise ValueError(
                    f"TRACE requires {name}={expected}; this checkpoint is incompatible."
                )
    for name in _ZERO_FIELDS:
        if name in result and result.pop(name) != 0:
            raise ValueError(f"TRACE requires {name}=0; this checkpoint is incompatible.")
    # This field never participates in the full-method slot update.
    result.pop("num_memory_slots", None)
    if "observation.anchor_image" in result.get("input_features", {}):
        raise ValueError("TRACE checkpoints must contain regular camera observations only.")
    return result


class TraceConfigMixin:
    """Read-only architecture facts used by the dataset and runtime adapters."""

    @property
    def use_path_signature(self) -> bool:
        return True

    @property
    def use_delta_signature(self) -> bool:
        return True

    @property
    def use_prefix_sequence_training(self) -> bool:
        return True

    @property
    def use_visual_prefix_memory(self) -> bool:
        return True

    @property
    def use_signature_indexed_slot_memory(self) -> bool:
        return True

    @property
    def use_memory_conditioned_encoder_film(self) -> bool:
        return True

    @property
    def slot_memory_use_delta_routing(self) -> bool:
        return True

    @property
    def slot_memory_use_softmax_routing(self) -> bool:
        return True

    @property
    def slot_memory_use_readout_pooling(self) -> bool:
        return True

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **kwargs: Any):
        """Load a local or Hub config, validating historical architecture fields."""
        import draccus
        from huggingface_hub import hf_hub_download

        path = Path(pretrained_name_or_path).expanduser()
        if path.is_dir():
            config_path = path / "config.json"
        elif path.is_file():
            config_path = path
        else:
            download_kwargs = {
                key: kwargs[key]
                for key in ("force_download", "token", "cache_dir", "local_files_only", "revision")
                if key in kwargs
            }
            config_path = Path(
                hf_hub_download(str(pretrained_name_or_path), "config.json", **download_kwargs)
            )
        payload = normalize_trace_config(json.loads(config_path.read_text(encoding="utf-8")))
        payload.pop("type", None)
        with TemporaryDirectory(prefix="trace-config-") as temporary:
            normalized_path = Path(temporary) / "config.json"
            normalized_path.write_text(json.dumps(payload), encoding="utf-8")
            with draccus.config_type("json"):
                return draccus.parse(cls, normalized_path, args=kwargs.get("cli_overrides", []))
