#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from .prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PREFIX_IMAGES_PREFIX,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_MASK_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PREFIX_STATE_KEY,
    PATH_SIGNATURE_KEY,
    is_prefix_image_key,
)


@PreTrainedConfig.register_subclass("streaming_act")
@dataclass
class StreamingACTConfig(TraceConfigMixin, PreTrainedConfig):
    """Full TRACE with an ACT action decoder.

    Path signatures, their increments, prefix reconstruction and signature-indexed
    slot memory are always present. Configure sensor features, history budgets
    and model dimensions for the target task. Architecture facts are read-only
    properties rather than experiment switches.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    prefix_train_max_steps: int = 32
    prefix_frame_stride: int = 1
    prefix_pad_value: float = 0.0
    slot_memory_num_slots: int = 4
    slot_memory_routing_hidden_dim: int = 512
    slot_memory_identity_scale: float = 0.1
    slot_memory_routing_temperature: float = 1.0

    # Streaming path-signature history module.
    history_length: int = 10
    # Signature output dimensionality is typically a function of channels and truncation depth.
    # Set to 0 to infer automatically in the model implementation.
    signature_dim: int = 0
    signature_depth: int = 3
    signature_hidden_dim: int = 512
    signature_dropout: float = 0.1
    pre_normalized_observation_keys: tuple[str, ...] = field(default_factory=tuple)

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.history_length <= 0:
            raise ValueError(f"`history_length` must be > 0. Got {self.history_length}.")
        if self.signature_dim < 0:
            raise ValueError(f"`signature_dim` must be >= 0. Got {self.signature_dim}.")
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
                "`slot_memory_num_slots` must be > 0 when "
                "TRACE slot memory. "
                f"Got {self.slot_memory_num_slots}."
            )
        if self.slot_memory_routing_hidden_dim <= 0:
            raise ValueError(
                "`slot_memory_routing_hidden_dim` must be > 0 when "
                "TRACE slot memory. "
                f"Got {self.slot_memory_routing_hidden_dim}."
            )
        if self.slot_memory_identity_scale < 0.0:
            raise ValueError(
                "`slot_memory_identity_scale` must be >= 0.0 when "
                "TRACE slot memory. "
                f"Got {self.slot_memory_identity_scale}."
            )
        if self.slot_memory_routing_temperature <= 0.0:
            raise ValueError(
                "`slot_memory_routing_temperature` must be > 0.0 when "
                "TRACE slot memory. "
                f"Got {self.slot_memory_routing_temperature}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.visual_observation_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )
        path_signature_feature = self.path_signature_feature
        if path_signature_feature is None:
            raise ValueError(
                f"TRACE signature conditioning requires input feature `{PATH_SIGNATURE_KEY}`."
            )
        expected_signature_shape = (self.signature_dim,)
        if tuple(path_signature_feature.shape) != expected_signature_shape:
            raise ValueError(
                "Path-signature feature shape mismatch. "
                f"Expected {expected_signature_shape}, got "
                f"{tuple(path_signature_feature.shape)}."
            )
        if not self.robot_state_feature:
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
                "Prefix state feature shape mismatch. "
                f"Expected {expected_prefix_state_shape}, got {tuple(prefix_state_feature.shape)}."
            )

        prefix_mask_feature = self.prefix_mask_feature
        if prefix_mask_feature is None:
            raise ValueError(
                f"TRACE prefix reconstruction requires input feature `{PREFIX_MASK_KEY}`."
            )
        if tuple(prefix_mask_feature.shape) != (self.prefix_train_max_steps,):
            raise ValueError(
                "Prefix mask feature shape mismatch. "
                f"Expected {(self.prefix_train_max_steps,)}, got {tuple(prefix_mask_feature.shape)}."
            )

        if not self.prefix_image_features:
            raise ValueError(
                "TRACE prefix reconstruction requires at least one "
                "`observation.prefix_images.*` feature."
            )

        if len(self.prefix_image_features) != len(self.visual_observation_features):
            raise ValueError(
                "Prefix image feature count must match the number of regular observation cameras. "
                f"Got prefix={len(self.prefix_image_features)} vs "
                f"current={len(self.visual_observation_features)}."
            )

        current_image_shapes = {}
        for key, feature in self.visual_observation_features.items():
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
            suffix = prefix_key.removeprefix(PREFIX_IMAGES_PREFIX)
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
        prefix_signature_feature = self.prefix_path_signature_feature
        if prefix_signature_feature is None:
            raise ValueError(
                "TRACE prefix reconstruction requires input feature "
                f"`{PREFIX_PATH_SIGNATURE_KEY}` when TRACE signature conditioning."
            )
        expected_prefix_signature_shape = (
            self.prefix_train_max_steps,
            self.signature_dim,
        )
        if tuple(prefix_signature_feature.shape) != expected_prefix_signature_shape:
            raise ValueError(
                "Prefix path-signature feature shape mismatch. "
                f"Expected {expected_prefix_signature_shape}, got "
                f"{tuple(prefix_signature_feature.shape)}."
            )
        delta_signature_feature = self.delta_signature_feature
        if delta_signature_feature is None:
            raise ValueError(
                f"TRACE signature increments requires input feature `{DELTA_SIGNATURE_KEY}`."
            )
        expected_delta_signature_shape = (self.signature_dim,)
        if tuple(delta_signature_feature.shape) != expected_delta_signature_shape:
            raise ValueError(
                "Delta-signature feature shape mismatch. "
                f"Expected {expected_delta_signature_shape}, got "
                f"{tuple(delta_signature_feature.shape)}."
            )

        prefix_delta_signature_feature = self.prefix_delta_signature_feature
        if prefix_delta_signature_feature is None:
            raise ValueError(
                "TRACE prefix reconstruction requires input feature "
                f"`{PREFIX_DELTA_SIGNATURE_KEY}` when TRACE signature increments."
            )
        expected_prefix_delta_signature_shape = (
            self.prefix_train_max_steps,
            self.signature_dim,
        )
        if tuple(prefix_delta_signature_feature.shape) != expected_prefix_delta_signature_shape:
            raise ValueError(
                "Prefix delta-signature feature shape mismatch. "
                f"Expected {expected_prefix_delta_signature_shape}, got "
                f"{tuple(prefix_delta_signature_feature.shape)}."
            )
        if not self.visual_observation_features:
            raise ValueError(
                "TRACE visual memory requires at least one regular observation image feature."
            )
        if not self.robot_state_feature:
            raise ValueError("TRACE visual memory requires `observation.state`.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def visual_observation_features(self) -> dict:
        return {key: ft for key, ft in self.image_features.items() if not is_prefix_image_key(key)}

    @property
    def prefix_state_feature(self):
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_STATE_KEY)

    @property
    def prefix_path_signature_feature(self):
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_PATH_SIGNATURE_KEY)

    @property
    def path_signature_feature(self):
        if not self.input_features:
            return None
        return self.input_features.get(PATH_SIGNATURE_KEY)

    @property
    def delta_signature_feature(self):
        if not self.input_features:
            return None
        return self.input_features.get(DELTA_SIGNATURE_KEY)

    @property
    def prefix_delta_signature_feature(self):
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_DELTA_SIGNATURE_KEY)

    @property
    def prefix_mask_feature(self):
        if not self.input_features:
            return None
        return self.input_features.get(PREFIX_MASK_KEY)

    @property
    def prefix_image_features(self) -> dict:
        return {key: ft for key, ft in self.image_features.items() if is_prefix_image_key(key)}

    @property
    def active_visual_prefix_memory_num_slots(self) -> int:
        return int(self.slot_memory_num_slots)
