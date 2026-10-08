#!/usr/bin/env python3
"""Run a tiny TRACE forward pass on CPU with synthetic observations.

Run from any directory with ``python /path/to/trace/examples/smoke_trace.py``.
This checks model wiring, not learned behavior or paper performance. It downloads
no weights and needs no dataset, simulator, or robot. Path signatures are computed
by signatory from a synthetic 2D trajectory; images and model weights are random.
"""

from __future__ import annotations

import sys
from pathlib import Path

import signatory
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "policy" / "lerobot_policy_streaming_act" / "src"))

from policy_imports import ensure_lerobot_policy_imports

ensure_lerobot_policy_imports("streaming_act")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_streaming_act.configuration_streaming_act import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    StreamingACTConfig,
)
from lerobot_policy_streaming_act.modeling_streaming_act import StreamingACTPolicy
from lerobot_policy_streaming_act.prefix_sequence import (
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    build_prefix_sequence_input_features,
    prefix_image_key_from_camera_key,
)


def build_smoke_example() -> tuple[StreamingACTPolicy, dict[str, torch.Tensor]]:
    """Build a small full-method policy and a deterministic synthetic batch."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    batch_size, prefix_steps, state_dim, signature_depth = 2, 4, 2, 2
    signature_dim = signatory.signature_channels(state_dim, signature_depth)
    image_key = "observation.images.main"

    base_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        PATH_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(signature_dim,)),
        DELTA_SIGNATURE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(signature_dim,)),
    }
    config = StreamingACTConfig(
        device="cpu",
        input_features=build_prefix_sequence_input_features(
            base_input_features=base_features,
            prefix_train_max_steps=prefix_steps,
        ),
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(state_dim,))},
        prefix_train_max_steps=prefix_steps,
        prefix_frame_stride=1,
        slot_memory_num_slots=2,
        slot_memory_routing_temperature=0.5,
        signature_dim=signature_dim,
        signature_depth=signature_depth,
        signature_hidden_dim=16,
        signature_dropout=0.0,
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        dim_feedforward=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
        pretrained_backbone_weights=None,
    )
    policy = StreamingACTPolicy(config).cpu().eval()

    states = torch.randn(batch_size, prefix_steps, state_dim).cumsum(dim=1)
    # Match data/process_dataset.py: the first point is the signature basepoint,
    # and delta signatures are adjacent differences, with a zero first delta.
    signatures = signatory.signature(
        states, depth=signature_depth, stream=True, basepoint=states[:, 0, :]
    )
    delta_signatures = torch.zeros_like(signatures)
    delta_signatures[:, 1:] = signatures[:, 1:] - signatures[:, :-1]
    images = torch.rand(batch_size, prefix_steps, 3, 32, 32)
    valid_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    last_valid = valid_mask.sum(dim=1) - 1
    rows = torch.arange(batch_size)
    batch = {
        "observation.state": states[rows, last_valid],
        image_key: images[rows, last_valid],
        PATH_SIGNATURE_KEY: signatures[rows, last_valid],
        DELTA_SIGNATURE_KEY: delta_signatures[rows, last_valid],
        PREFIX_STATE_KEY: states.masked_fill(~valid_mask[..., None], 0),
        PREFIX_PATH_SIGNATURE_KEY: signatures.masked_fill(~valid_mask[..., None], 0),
        PREFIX_DELTA_SIGNATURE_KEY: delta_signatures.masked_fill(~valid_mask[..., None], 0),
        PREFIX_MASK_KEY: valid_mask,
        prefix_image_key_from_camera_key(image_key): images.masked_fill(
            ~valid_mask[..., None, None, None], 0
        ),
    }

    return policy, batch


def main() -> None:
    """Check actions and slot-memory diagnostics for a small synthetic batch."""
    policy, batch = build_smoke_example()
    with torch.inference_mode():
        # Exercise the complete masked prefix, as in the training forward path.
        # predict_action_chunk() instead updates the online memory by one frame.
        actions, _latent = policy.model(policy._prepare_observation_batch(batch))

    assert actions.shape == (2, policy.config.chunk_size, 2), actions.shape
    assert actions.device.type == "cpu"
    assert torch.isfinite(actions).all(), "Model produced non-finite actions."
    log_stats = policy.model.get_visual_prefix_memory_log_stats()
    assert "slot_memory/routing_weight/slot_0" in log_stats
    assert "slot_memory/routing_weight/slot_1" in log_stats

    print("TRACE CPU smoke passed (synthetic inputs; randomly initialized model).")
    print(f"Signatures: 2D path, depth={policy.config.signature_depth}, channels={policy.config.signature_dim}")
    print(f"Actions: shape={tuple(actions.shape)}, finite={bool(torch.isfinite(actions).all())}")
    print("Slot memory: 2 slots; path/delta routing and encoder FiLM enabled.")


if __name__ == "__main__":
    main()
