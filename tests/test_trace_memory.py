"""Full TRACE memory contracts on small, deterministic CPU observations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "policy" / "lerobot_policy_prism_diffusion" / "src"))

from examples import smoke_trace
from lerobot_policy_streaming_act.prefix_sequence import select_prefix_positions


@pytest.fixture
def example():
    return smoke_trace.build_smoke_example()


def _observation_at(batch: dict[str, torch.Tensor], step: int) -> dict[str, torch.Tensor]:
    """Read one episode's current observation without providing its future frames."""
    image_key = "observation.images.main"
    prefix_image_key = smoke_trace.prefix_image_key_from_camera_key(image_key)
    return {
        "observation.state": batch[smoke_trace.PREFIX_STATE_KEY][:1, step],
        image_key: batch[prefix_image_key][:1, step],
        smoke_trace.PATH_SIGNATURE_KEY: batch[smoke_trace.PREFIX_PATH_SIGNATURE_KEY][:1, step],
        smoke_trace.DELTA_SIGNATURE_KEY: batch[smoke_trace.PREFIX_DELTA_SIGNATURE_KEY][:1, step],
    }


def test_bounded_prefix_keeps_episode_start_and_recent_frames() -> None:
    positions = select_prefix_positions(
        sequence_length=10,
        prefix_train_max_steps=4,
        prefix_frame_stride=2,
    )
    assert positions == [0, 6, 8, 9]


def test_full_trace_forward_is_finite(example) -> None:
    policy, batch = example
    with torch.inference_mode():
        actions, _latent = policy.model(policy._prepare_observation_batch(batch))

    assert actions.shape == (2, 2, 2)
    assert actions.device.type == "cpu"
    assert torch.isfinite(actions).all()
    stats = policy.model.get_visual_prefix_memory_log_stats()
    assert "slot_memory/routing_weight/slot_0" in stats
    assert "slot_memory/routing_weight/slot_1" in stats


def test_masked_future_frames_do_not_change_actions(example) -> None:
    policy, batch = example
    changed_batch = {key: value.clone() for key, value in batch.items()}
    prefix_keys = (
        smoke_trace.PREFIX_STATE_KEY,
        smoke_trace.PREFIX_PATH_SIGNATURE_KEY,
        smoke_trace.PREFIX_DELTA_SIGNATURE_KEY,
        smoke_trace.prefix_image_key_from_camera_key("observation.images.main"),
    )
    # Episode 1 ends after two frames. Its masked suffix must not affect memory.
    for key in prefix_keys:
        changed_batch[key][1, 2:] = 100.0 * torch.randn_like(changed_batch[key][1, 2:])

    with torch.inference_mode():
        before, _latent = policy.model(policy._prepare_observation_batch(batch))
        after, _latent = policy.model(policy._prepare_observation_batch(changed_batch))

    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)


def test_online_memory_matches_causal_prefix_forward(example) -> None:
    policy, batch = example
    with torch.inference_mode():
        prefix_actions, _latent = policy.model(policy._prepare_observation_batch(batch))
        policy.reset()
        for step in range(batch[smoke_trace.PREFIX_MASK_KEY].shape[1]):
            online_actions = policy.predict_action_chunk(_observation_at(batch, step))

    torch.testing.assert_close(prefix_actions[:1], online_actions, atol=1e-5, rtol=1e-4)
    assert policy.get_visual_prefix_memory_debug_stats()["update_count"] == 4


def test_episode_reset_restores_initial_prediction(example) -> None:
    policy, batch = example
    initial_observation = _observation_at(batch, 0)
    with torch.inference_mode():
        initial_actions = policy.predict_action_chunk(initial_observation)
        policy.select_action(_observation_at(batch, 1))
        assert policy.get_visual_prefix_memory_debug_stats()["initialized"]

        policy.reset()
        reset_stats = policy.get_visual_prefix_memory_debug_stats()
        assert not reset_stats["initialized"]
        assert reset_stats["update_count"] == 0
        assert len(policy._action_queue) == 0
        new_episode_actions = policy.predict_action_chunk(initial_observation)

    torch.testing.assert_close(initial_actions, new_episode_actions, atol=1e-6, rtol=1e-5)


def test_diffusion_training_sampling_and_episode_reset(example) -> None:
    from lerobot_policy_prism_diffusion.configuration_diffusion import PrismDiffusionConfig
    from lerobot_policy_prism_diffusion.modeling_diffusion import PrismDiffusionPolicy

    act_policy, batch = example
    config = PrismDiffusionConfig(
        device="cpu",
        input_features=act_policy.config.input_features,
        output_features=act_policy.config.output_features,
        n_obs_steps=2,
        horizon=4,
        n_action_steps=1,
        down_dims=(32, 32),
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        pretrained_backbone_weights=None,
        num_train_timesteps=2,
        num_inference_steps=2,
        history_length=4,
        signature_dim=6,
        signature_depth=2,
        signature_hidden_dim=16,
        signature_dropout=0.0,
        prefix_train_max_steps=4,
        slot_memory_num_slots=2,
        slot_memory_routing_hidden_dim=16,
        prism_adapter_hidden_dim=32,
    )
    policy = PrismDiffusionPolicy(config).cpu().train()
    last_valid = batch[smoke_trace.PREFIX_MASK_KEY].sum(dim=1) - 1
    recent_positions = (last_valid[:, None] - torch.tensor([1, 0])).clamp_min(0)
    rows = torch.arange(2)[:, None]
    image_key = "observation.images.main"
    prefix_image_key = smoke_trace.prefix_image_key_from_camera_key(image_key)
    training_batch = dict(batch)
    training_batch["observation.state"] = batch[smoke_trace.PREFIX_STATE_KEY][rows, recent_positions]
    training_batch[image_key] = batch[prefix_image_key][rows, recent_positions]
    training_batch["action"] = torch.randn(2, config.horizon, 2)
    training_batch["action_is_pad"] = torch.zeros(2, config.horizon, dtype=torch.bool)

    loss, _ = policy(training_batch)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    adapter_gradient = policy.diffusion.prism_adapter[-1].weight.grad
    assert adapter_gradient is not None
    assert torch.count_nonzero(adapter_gradient), "The loss must train the TRACE conditioning adapter."

    policy.eval()
    with torch.inference_mode():
        torch.manual_seed(42)
        first_action = policy.select_action(_observation_at(batch, 0))
        policy.select_action(_observation_at(batch, 1))
        assert first_action.shape == (1, 2)
        assert torch.isfinite(first_action).all()
        assert policy.get_prism_memory_debug_stats()["update_count"] == 2

        policy.reset()
        assert not policy.get_prism_memory_debug_stats()["initialized"]
        assert policy.get_prism_memory_debug_stats()["update_count"] == 0
        assert all(len(queue) == 0 for queue in policy._queues.values())
        torch.manual_seed(42)
        fresh_action = policy.select_action(_observation_at(batch, 0))

    torch.testing.assert_close(first_action, fresh_action, atol=1e-6, rtol=1e-5)
