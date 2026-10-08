"""Historical checkpoints must describe the published TRACE architecture."""

import pytest

from lerobot_policy_streaming_act.trace_config import normalize_trace_config


def test_full_checkpoint_fields_are_normalized_without_mutating_input() -> None:
    payload = {
        "type": "streaming_act",
        "use_path_signature": True,
        "use_delta_signature": True,
        "use_prefix_sequence_training": True,
        "use_visual_prefix_memory": True,
        "use_signature_indexed_slot_memory": True,
        "use_memory_conditioned_encoder_film": True,
        "slot_memory_use_delta_routing": True,
        "slot_memory_use_softmax_routing": True,
        "slot_memory_use_readout_pooling": True,
        "use_signature_conditioned_visual_prefix_memory": False,
        "use_first_frame_anchor": False,
        "use_first_frame_anchor_in_slot_routing": False,
        "slot_memory_balance_loss_coef": 0.0,
        "slot_memory_entropy_loss_coef": 0.0,
        "slot_memory_consistency_loss_coef": 0.0,
        "num_memory_slots": 1,
        "slot_memory_num_slots": 4,
        "future_metadata": "preserved for the caller",
    }

    normalized = normalize_trace_config(payload)

    assert normalized == {
        "type": "streaming_act",
        "slot_memory_num_slots": 4,
        "future_metadata": "preserved for the caller",
    }
    assert payload["use_path_signature"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("use_path_signature", False),
        ("use_delta_signature", False),
        ("use_prefix_sequence_training", False),
        ("use_visual_prefix_memory", False),
        ("use_signature_indexed_slot_memory", False),
        ("use_memory_conditioned_encoder_film", False),
        ("slot_memory_use_delta_routing", False),
        ("slot_memory_use_softmax_routing", False),
        ("slot_memory_use_readout_pooling", False),
        ("use_signature_conditioned_visual_prefix_memory", True),
        ("use_first_frame_anchor", True),
        ("use_first_frame_anchor_in_slot_routing", True),
        ("slot_memory_balance_loss_coef", 0.1),
        ("slot_memory_entropy_loss_coef", 0.1),
        ("slot_memory_consistency_loss_coef", 0.1),
        ("use_path_signature", "true"),
    ],
)
def test_incompatible_checkpoint_is_rejected(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        normalize_trace_config({name: value})


def test_new_checkpoint_keeps_architecture_dimensions() -> None:
    payload = {"signature_dim": 6, "slot_memory_num_slots": 2, "prefix_train_max_steps": 4}

    assert normalize_trace_config(payload) == payload
