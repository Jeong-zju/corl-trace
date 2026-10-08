#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TRACE modifications (2026): causal signatures and memory, with full-method specialization.
from dataclasses import dataclass, field

from lerobot_policy_streaming_act.trace_config import TraceConfigMixin

try:
    from policy_imports import install_lerobot_policies_namespace_shim
except ModuleNotFoundError:

    def install_lerobot_policies_namespace_shim() -> None:
        return None


install_lerobot_policies_namespace_shim()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot_policy_streaming_act.prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    is_prefix_image_key,
)


@PreTrainedConfig.register_subclass("prism_diffusion")
@dataclass
class PrismDiffusionConfig(TraceConfigMixin, PreTrainedConfig):
    """Full TRACE with a diffusion action decoder.

    Path signatures, their increments, prefix reconstruction and signature-indexed
    slot memory are always present. Configure sensor features, history budgets
    and model dimensions for the target task. Architecture facts are read-only
    properties rather than experiment switches.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Processor/runtime controls
    pre_normalized_observation_keys: tuple[str, ...] = field(default_factory=tuple)
    history_length: int = 0
    signature_dim: int = 0
    signature_depth: int = 3
    signature_hidden_dim: int = 512
    signature_dropout: float = 0.1
    prefix_train_max_steps: int = 32
    prefix_frame_stride: int = 1
    prefix_pad_value: float = 0.0
    slot_memory_num_slots: int = 4
    slot_memory_routing_hidden_dim: int = 512
    prism_adapter_hidden_dim: int = 512
    prism_adapter_zero_init: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(
                f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}."
            )
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # Explicitly disable cropping for resize+ratio path when crop_ratio == 1.0.
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )
        if self.history_length <= 0:
            raise ValueError(f"`history_length` must be > 0. Got {self.history_length}.")
        if self.signature_dim <= 0:
            raise ValueError(f"`signature_dim` must be > 0. Got {self.signature_dim}.")
        if self.signature_depth <= 0:
            raise ValueError(f"`signature_depth` must be > 0. Got {self.signature_depth}.")
        if self.signature_hidden_dim <= 0:
            raise ValueError(
                f"`signature_hidden_dim` must be > 0. Got {self.signature_hidden_dim}."
            )
        if not (0.0 <= self.signature_dropout <= 1.0):
            raise ValueError(
                f"`signature_dropout` must be in [0, 1]. Got {self.signature_dropout}."
            )
        if self.prefix_train_max_steps <= 0:
            raise ValueError(
                "`prefix_train_max_steps` must be > 0 when "
                f"TRACE prefix reconstruction. Got {self.prefix_train_max_steps}."
            )
        if self.prefix_frame_stride <= 0:
            raise ValueError(
                "`prefix_frame_stride` must be > 0 when "
                f"TRACE prefix reconstruction. Got {self.prefix_frame_stride}."
            )
        if self.slot_memory_num_slots <= 0:
            raise ValueError(
                "`slot_memory_num_slots` must be > 0 when PRISM memory is enabled. "
                f"Got {self.slot_memory_num_slots}."
            )
        if self.slot_memory_routing_hidden_dim <= 0:
            raise ValueError(
                "`slot_memory_routing_hidden_dim` must be > 0 when slot memory routing is enabled. "
                f"Got {self.slot_memory_routing_hidden_dim}."
            )
        if self.prism_adapter_hidden_dim <= 0:
            raise ValueError(
                f"`prism_adapter_hidden_dim` must be > 0. Got {self.prism_adapter_hidden_dim}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )
        path_signature_feature = self.path_signature_feature
        if path_signature_feature is None:
            raise ValueError(
                f"TRACE signature conditioning requires input feature `{PATH_SIGNATURE_KEY}`."
            )
        expected_shape = (self.signature_dim,)
        if tuple(path_signature_feature.shape) != expected_shape:
            raise ValueError(
                "Path-signature feature shape mismatch. "
                f"Expected {expected_shape}, got {tuple(path_signature_feature.shape)}."
            )
        delta_signature_feature = self.delta_signature_feature
        if delta_signature_feature is None:
            raise ValueError(
                f"TRACE signature increments requires input feature `{DELTA_SIGNATURE_KEY}`."
            )
        expected_shape = (self.signature_dim,)
        if tuple(delta_signature_feature.shape) != expected_shape:
            raise ValueError(
                "Delta-signature feature shape mismatch. "
                f"Expected {expected_shape}, got {tuple(delta_signature_feature.shape)}."
            )
        if self.robot_state_feature is None:
            raise ValueError(
                "TRACE prefix reconstruction requires a current `observation.state` feature."
            )
        prefix_state_feature = self.prefix_state_feature
        if prefix_state_feature is None:
            raise ValueError(
                f"TRACE prefix reconstruction requires input feature `{PREFIX_STATE_KEY}`."
            )
        expected_prefix_state_shape = (
            self.prefix_train_max_steps,
            self.robot_state_feature.shape[0],
        )
        if tuple(prefix_state_feature.shape) != expected_prefix_state_shape:
            raise ValueError(
                "Prefix-state feature shape mismatch. "
                f"Expected {expected_prefix_state_shape}, got {tuple(prefix_state_feature.shape)}."
            )
        prefix_mask_feature = self.prefix_mask_feature
        if prefix_mask_feature is None:
            raise ValueError(
                f"TRACE prefix reconstruction requires input feature `{PREFIX_MASK_KEY}`."
            )
        if tuple(prefix_mask_feature.shape) != (self.prefix_train_max_steps,):
            raise ValueError(
                "Prefix-mask feature shape mismatch. "
                f"Expected {(self.prefix_train_max_steps,)}, got {tuple(prefix_mask_feature.shape)}."
            )
        if not self.prefix_image_features:
            raise ValueError(
                "TRACE prefix reconstruction requires at least one "
                "`observation.prefix_images.*` feature."
            )
        if len(self.prefix_image_features) != len(self.image_features):
            raise ValueError(
                "Prefix image feature count must match the number of current observation cameras. "
                f"Got prefix={len(self.prefix_image_features)} vs current={len(self.image_features)}."
            )

        current_image_shapes: dict[str, tuple[int, ...]] = {}
        for key, feature in self.image_features.items():
            if key.startswith("observation.images."):
                suffix = key.removeprefix("observation.images.")
            elif key == "observation.image":
                suffix = "main"
            else:
                raise ValueError(
                    f"Unsupported current image feature name for prefix-sequence mode. Got `{key}`."
                )
            current_image_shapes[suffix] = tuple(feature.shape)

        for prefix_key, prefix_feature in self.prefix_image_features.items():
            suffix = prefix_key.removeprefix("observation.prefix_images.")
            current_shape = current_image_shapes.get(suffix)
            if current_shape is None:
                raise ValueError(
                    "Prefix image feature does not map to any regular observation image. "
                    f"Got prefix key `{prefix_key}`."
                )
            expected_prefix_shape = (self.prefix_train_max_steps, *current_shape)
            if tuple(prefix_feature.shape) != expected_prefix_shape:
                raise ValueError(
                    "Prefix image feature shape mismatch. "
                    f"Expected {expected_prefix_shape} for `{prefix_key}`, "
                    f"got {tuple(prefix_feature.shape)}."
                )
        prefix_path_signature_feature = self.prefix_path_signature_feature
        if prefix_path_signature_feature is None:
            raise ValueError(
                "TRACE prefix reconstruction requires input feature "
                f"`{PREFIX_PATH_SIGNATURE_KEY}` when TRACE signature conditioning."
            )
        expected_shape = (self.prefix_train_max_steps, self.signature_dim)
        if tuple(prefix_path_signature_feature.shape) != expected_shape:
            raise ValueError(
                "Prefix path-signature feature shape mismatch. "
                f"Expected {expected_shape}, got {tuple(prefix_path_signature_feature.shape)}."
            )
        prefix_delta_signature_feature = self.prefix_delta_signature_feature
        if prefix_delta_signature_feature is None:
            raise ValueError(
                "TRACE prefix reconstruction requires input feature "
                f"`{PREFIX_DELTA_SIGNATURE_KEY}` when TRACE signature increments."
            )
        expected_shape = (self.prefix_train_max_steps, self.signature_dim)
        if tuple(prefix_delta_signature_feature.shape) != expected_shape:
            raise ValueError(
                "Prefix delta-signature feature shape mismatch. "
                f"Expected {expected_shape}, got {tuple(prefix_delta_signature_feature.shape)}."
            )
        if not self.image_features:
            raise ValueError(
                "TRACE visual memory requires at least one regular observation image feature."
            )
        if self.robot_state_feature is None:
            raise ValueError("TRACE visual memory requires `observation.state`.")

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is FeatureType.VISUAL and not key.startswith("observation.prefix_images.")
        }

    @property
    def path_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PATH_SIGNATURE_KEY)

    @property
    def delta_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(DELTA_SIGNATURE_KEY)

    @property
    def prefix_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_STATE_KEY)

    @property
    def prefix_path_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_PATH_SIGNATURE_KEY)

    @property
    def prefix_delta_signature_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_DELTA_SIGNATURE_KEY)

    @property
    def prefix_mask_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_MASK_KEY)

    @property
    def prefix_image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if is_prefix_image_key(key)}

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
