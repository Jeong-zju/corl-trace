from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SIGNATURE_CAPABLE_POLICIES = frozenset({"streaming_act", "prism_diffusion"})


@dataclass(frozen=True)
class PolicyCapabilityFlags:
    use_path_signature: bool = field(default=True, init=False)
    use_delta_signature: bool = field(default=True, init=False)
    use_prefix_sequence_training: bool = field(default=True, init=False)
    use_visual_prefix_memory: bool = field(default=True, init=False)
    use_signature_indexed_slot_memory: bool = field(default=True, init=False)

    @property
    def build_explicit_prefix_eval_inputs(self) -> bool:
        return False


def policy_supports_signature_features(policy_name: str | None) -> bool:
    return str(policy_name).strip() in SIGNATURE_CAPABLE_POLICIES


def resolve_policy_capability_flags(cfg: Any) -> PolicyCapabilityFlags:
    return PolicyCapabilityFlags()


def get_visual_memory_debug_stats(policy: Any) -> dict[str, Any] | None:
    for getter_name in (
        "get_visual_prefix_memory_debug_stats",
        "get_prism_memory_debug_stats",
    ):
        getter = getattr(policy, getter_name, None)
        if callable(getter):
            stats = getter()
            if isinstance(stats, dict):
                return dict(stats)
    return None
