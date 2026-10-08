"""Architecture constants and capacity defaults for the public TRACE entry points."""

FULL_TRACE_DEFAULTS = {
    "collision_mode": "reject",
    "use_path_signature": True,
    "use_delta_signature": True,
    "use_prefix_sequence_training": True,
    "use_visual_prefix_memory": True,
    "use_signature_indexed_slot_memory": True,
    "use_memory_conditioned_encoder_film": True,
    "slot_memory_use_delta_routing": True,
    "slot_memory_use_softmax_routing": True,
    "slot_memory_use_readout_pooling": True,
    "prism_adapter_zero_init": True,
    "use_signature_conditioned_visual_prefix_memory": False,
    "use_first_frame_anchor": False,
    "use_first_frame_anchor_in_slot_routing": False,
    "build_explicit_prefix_eval_inputs": False,
    "slot_memory_balance_loss_coef": 0.0,
    "slot_memory_entropy_loss_coef": 0.0,
    "slot_memory_consistency_loss_coef": 0.0,
    "num_memory_slots": 1,
    "signature_backend": "signatory",
}

TRACE_CAPACITY_DEFAULTS = {
    "prefix_train_max_steps": 32,
    "prefix_frame_stride": 4,
    "prefix_pad_value": 0.0,
    "slot_memory_num_slots": 4,
    "slot_memory_routing_hidden_dim": 512,
    "slot_memory_identity_scale": 0.1,
    "slot_memory_routing_temperature": 1.0,
    "history_length": 0,
    "signature_dim": 0,
    "signature_depth": 3,
    "signature_hidden_dim": 512,
    "signature_dropout": 0.1,
    "prism_adapter_hidden_dim": 512,
}


def validate_trace_defaults(values):
    """Reject YAML settings that would remove a TRACE component."""
    for key, expected in FULL_TRACE_DEFAULTS.items():
        if key in values and values[key] != expected:
            raise ValueError(f"Full TRACE requires {key}={expected!r}; got {values[key]!r}.")
    return values
