from __future__ import annotations
from collections import deque
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree as ET
import numpy as np

try:
    import mujoco
except Exception as exc:
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None
from eval_helpers import (
    build_eval_observation,
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    resolve_signature_backend,
    write_summary,
)
from policy_capabilities import get_visual_memory_debug_stats, resolve_policy_capability_flags
from policy_defaults import load_policy_mode_defaults

ENV_NAME = "panda_route"
DEFAULT_IMAGE_SIZE = 128
DEFAULT_RANDOM_SEED = 17
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
DEFAULT_VIDEO_IMAGE_SIZE = 128
DEFAULT_STEP_PENALTY = -0.01
DEFAULT_GOAL_REWARD = 1.0
DEFAULT_START_RANDOMIZE = False
DEFAULT_PANDA_DESCRIPTION_VARIANT = "panda_nohand"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MUJOCO_MENAGERIE_ROOT = WORKSPACE_ROOT / "mujoco_menagerie"
LOCAL_FRANKA_PANDA_ROOT = LOCAL_MUJOCO_MENAGERIE_ROOT / "franka_emika_panda"
ROBOT_DESCRIPTIONS_CACHE_ROOT = WORKSPACE_ROOT / ".robot_descriptions_cache"
FRANKA_ARM_JOINT_NAMES = tuple((f"joint{index}" for index in range(1, 8)))
LEGACY_ARM_JOINT_NAMES = tuple((f"panda_joint{index}" for index in range(1, 8)))
FRANKA_FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
FRANKA_HOME_QPOS = np.asarray([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)
LEGACY_HOME_QPOS = np.asarray([0.0, 0.4, 0.0, -1.8, 0.0, 1.6, 0.0], dtype=np.float64)
FRANKA_OPEN_GRIPPER_QPOS = 0.04
TASK_ID_TO_START_NAME = {0: "S1", 1: "S2"}
TASK_ID_TO_BRANCH_NAME = {0: "H1", 1: "L1"}
TASK_ID_TO_GOAL_NAME = {0: "G1", 1: "G2"}
TASK_ID_VALUES = tuple(sorted(TASK_ID_TO_GOAL_NAME))
SCHEMATIC_BACKGROUND_COLOR = (244, 239, 228)
SCHEMATIC_OBSTACLE_COLOR = (68, 68, 68)
SCHEMATIC_START_COLOR = (76, 120, 168)
SCHEMATIC_SHARED_COLOR = (120, 120, 120)
SCHEMATIC_H1_COLOR = (44, 160, 44)
SCHEMATIC_L1_COLOR = (148, 103, 189)
SCHEMATIC_GOAL_COLOR_BY_NAME = {"G1": (27, 158, 119), "G2": (217, 95, 2)}
SCHEMATIC_CURSOR_OUTER = (255, 255, 255)
SCHEMATIC_CURSOR_INNER = (240, 70, 70)


@dataclass(frozen=True)
class RectangleRegion:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin >= self.xmax:
            raise ValueError(f"{self.name}: xmin must be smaller than xmax.")
        if self.ymin >= self.ymax:
            raise ValueError(f"{self.name}: ymin must be smaller than ymax.")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass(frozen=True)
class RectangleObstacle(RectangleRegion):
    pass


@dataclass(frozen=True)
class GoalRegion(RectangleRegion):
    pass


@dataclass(frozen=True)
class SemanticRegion:
    name: str
    rectangles: tuple[RectangleRegion, ...]

    def contains_point(self, x: float, y: float) -> bool:
        return any((rectangle.contains_point(x, y) for rectangle in self.rectangles))


@dataclass(frozen=True)
class MapConfig:
    workspace: RectangleRegion
    start_room: SemanticRegion
    task_start_regions: tuple[RectangleRegion, ...]
    merge_region_m1: SemanticRegion
    middle_shared_region: SemanticRegion
    upper_hole_region: SemanticRegion
    lower_hole_region: SemanticRegion
    post_branch_shared_region: SemanticRegion
    merge_region_m2: SemanticRegion
    goal_regions: tuple[GoalRegion, ...]
    terminal_regions: tuple[GoalRegion, ...]
    free_space_rectangles: tuple[RectangleRegion, ...]
    obstacle_rectangles: tuple[RectangleObstacle, ...]


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    task_code: str
    start_region_name: str
    branch_region_name: str
    target_goal_name: str


@dataclass(frozen=True)
class TrajectoryPhaseAnnotation:
    phase_labels: tuple[str, ...]
    first_m1_index: int | None
    first_branch_index: int | None
    first_m2_index: int | None
    first_terminal_index: int | None
    first_terminal_phase: str | None


class SupportsUniform(Protocol):
    def uniform(self, a: float, b: float) -> float: ...


def build_obstacles_from_free_rectangles(
    workspace: RectangleRegion, free_rectangles: tuple[RectangleRegion, ...]
) -> tuple[RectangleObstacle, ...]:
    x_coords = sorted(
        {
            workspace.xmin,
            workspace.xmax,
            *(rect.xmin for rect in free_rectangles),
            *(rect.xmax for rect in free_rectangles),
        }
    )
    y_coords = sorted(
        {
            workspace.ymin,
            workspace.ymax,
            *(rect.ymin for rect in free_rectangles),
            *(rect.ymax for rect in free_rectangles),
        }
    )
    cell_free: list[tuple[float, float, float, float]] = []
    cell_obstacles: list[tuple[float, float, float, float]] = []
    for x0, x1 in zip(x_coords[:-1], x_coords[1:], strict=False):
        for y0, y1 in zip(y_coords[:-1], y_coords[1:], strict=False):
            if x1 <= x0 or y1 <= y0:
                continue
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            if any((rect.contains_point(cx, cy) for rect in free_rectangles)):
                cell_free.append((x0, x1, y0, y1))
            else:
                cell_obstacles.append((x0, x1, y0, y1))
    merged: list[list[float]] = []
    for x0, x1, y0, y1 in cell_obstacles:
        merged_existing = False
        for rect in merged:
            same_x = math.isclose(rect[0], x0) and math.isclose(rect[1], x1)
            touching_y = math.isclose(rect[3], y0) or math.isclose(rect[2], y1)
            overlap_y = not (y1 < rect[2] or y0 > rect[3])
            if same_x and (touching_y or overlap_y):
                rect[2] = min(rect[2], y0)
                rect[3] = max(rect[3], y1)
                merged_existing = True
                break
        if not merged_existing:
            merged.append([x0, x1, y0, y1])
    merged_second_pass: list[list[float]] = []
    for x0, x1, y0, y1 in merged:
        merged_existing = False
        for rect in merged_second_pass:
            same_y = math.isclose(rect[2], y0) and math.isclose(rect[3], y1)
            touching_x = math.isclose(rect[1], x0) or math.isclose(rect[0], x1)
            overlap_x = not (x1 < rect[0] or x0 > rect[1])
            if same_y and (touching_x or overlap_x):
                rect[0] = min(rect[0], x0)
                rect[1] = max(rect[1], x1)
                merged_existing = True
                break
        if not merged_existing:
            merged_second_pass.append([x0, x1, y0, y1])
    obstacles = tuple(
        (
            RectangleObstacle(
                name=f"obstacle_{index:02d}",
                xmin=float(x0),
                xmax=float(x1),
                ymin=float(y0),
                ymax=float(y1),
            )
            for index, (x0, x1, y0, y1) in enumerate(merged_second_pass)
        )
    )
    return obstacles


def build_task_spec(task_id: int) -> TaskSpec:
    if task_id not in TASK_ID_TO_GOAL_NAME:
        raise ValueError(f"Unsupported task_id={task_id}.")
    return TaskSpec(
        task_id=int(task_id),
        task_code=f"T{task_id}",
        start_region_name=TASK_ID_TO_START_NAME[task_id],
        branch_region_name=TASK_ID_TO_BRANCH_NAME[task_id],
        target_goal_name=TASK_ID_TO_GOAL_NAME[task_id],
    )


def build_default_map_config() -> MapConfig:
    workspace = RectangleRegion("workspace", 0.12, 0.78, -0.28, 0.28)
    start_room_rect = RectangleRegion("start_room_rect", 0.18, 0.31, -0.24, 0.24)
    s1 = RectangleRegion("S1", 0.18, 0.25, 0.1, 0.2)
    s2 = RectangleRegion("S2", 0.18, 0.25, -0.2, -0.1)
    m1 = RectangleRegion("M1", 0.31, 0.39, -0.05, 0.05)
    middle_shared = RectangleRegion("middle_shared_region_rect", 0.39, 0.43, -0.24, 0.24)
    h1 = RectangleRegion("H1", 0.43, 0.51, 0.1, 0.18)
    l1 = RectangleRegion("L1", 0.43, 0.51, -0.18, -0.1)
    post_branch_shared = RectangleRegion("post_branch_shared_region_rect", 0.51, 0.55, -0.24, 0.24)
    m2 = RectangleRegion("M2", 0.55, 0.63, -0.05, 0.05)
    goal_room = RectangleRegion("goal_room_rect", 0.63, 0.74, -0.24, 0.24)
    g1 = GoalRegion("G1", 0.66, 0.74, 0.1, 0.2)
    g2 = GoalRegion("G2", 0.66, 0.74, -0.2, -0.1)
    free_space_rectangles = (
        start_room_rect,
        m1,
        middle_shared,
        h1,
        l1,
        post_branch_shared,
        m2,
        goal_room,
    )
    obstacle_rectangles = build_obstacles_from_free_rectangles(
        workspace=workspace, free_rectangles=free_space_rectangles
    )
    return MapConfig(
        workspace=workspace,
        start_room=SemanticRegion("start_room", (start_room_rect,)),
        task_start_regions=(s1, s2),
        merge_region_m1=SemanticRegion("M1", (m1,)),
        middle_shared_region=SemanticRegion("middle_shared_region", (middle_shared,)),
        upper_hole_region=SemanticRegion("H1", (h1,)),
        lower_hole_region=SemanticRegion("L1", (l1,)),
        post_branch_shared_region=SemanticRegion(
            "post_branch_shared_region", (post_branch_shared,)
        ),
        merge_region_m2=SemanticRegion("M2", (m2,)),
        goal_regions=(g1, g2),
        terminal_regions=(g1, g2),
        free_space_rectangles=free_space_rectangles,
        obstacle_rectangles=obstacle_rectangles,
    )


def get_task_start_region(task_id: int, config: MapConfig | None = None) -> RectangleRegion:
    resolved_config = build_default_map_config() if config is None else config
    if task_id == 0:
        return resolved_config.task_start_regions[0]
    if task_id == 1:
        return resolved_config.task_start_regions[1]
    raise ValueError(f"Unsupported task_id={task_id}.")


def get_goal_region_for_state(
    x: float, y: float, config: MapConfig | None = None
) -> GoalRegion | None:
    resolved_config = build_default_map_config() if config is None else config
    for goal_region in resolved_config.goal_regions:
        if goal_region.contains_point(x, y):
            return goal_region
    return None


def is_state_valid(x: float, y: float, config: MapConfig | None = None) -> bool:
    resolved_config = build_default_map_config() if config is None else config
    if not resolved_config.workspace.contains_point(x, y):
        return False
    return any((rect.contains_point(x, y) for rect in resolved_config.free_space_rectangles))


def sample_valid_state_in_region(
    region: RectangleRegion,
    rng: SupportsUniform,
    config: MapConfig | None = None,
    max_attempts: int = 2000,
) -> tuple[float, float]:
    resolved_config = build_default_map_config() if config is None else config
    for _ in range(max_attempts):
        x = float(rng.uniform(region.xmin, region.xmax))
        y = float(rng.uniform(region.ymin, region.ymax))
        if is_state_valid(x, y, config=resolved_config):
            return (x, y)
    raise RuntimeError(f"Could not sample a valid state in region {region.name}.")


def get_phase_name(x: float, y: float, config: MapConfig | None = None) -> str:
    resolved_config = build_default_map_config() if config is None else config
    if not resolved_config.workspace.contains_point(x, y):
        return "out_of_bounds"
    if not is_state_valid(x, y, config=resolved_config):
        return "obstacle"
    if resolved_config.task_start_regions[0].contains_point(x, y):
        return "S1"
    if resolved_config.task_start_regions[1].contains_point(x, y):
        return "S2"
    for goal_region in resolved_config.goal_regions:
        if goal_region.contains_point(x, y):
            return goal_region.name
    if resolved_config.merge_region_m1.contains_point(x, y):
        return "M1"
    if resolved_config.upper_hole_region.contains_point(x, y):
        return "H1"
    if resolved_config.lower_hole_region.contains_point(x, y):
        return "L1"
    if resolved_config.merge_region_m2.contains_point(x, y):
        return "M2"
    if resolved_config.middle_shared_region.contains_point(x, y):
        return "middle_shared_region"
    if resolved_config.post_branch_shared_region.contains_point(x, y):
        return "post_branch_shared_region"
    if resolved_config.start_room.contains_point(x, y):
        return "start_room"
    return "free_space_other"


def _first_index_of_phase(phase_labels: tuple[str, ...], target_phase: str) -> int | None:
    for index, phase_label in enumerate(phase_labels):
        if phase_label == target_phase:
            return index
    return None


def annotate_trajectory_phases(
    path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig | None = None,
) -> TrajectoryPhaseAnnotation:
    resolved_config = build_default_map_config() if config is None else config
    phase_labels = tuple((get_phase_name(x, y, config=resolved_config) for x, y in path))
    terminal_phase_names = {goal_region.name for goal_region in resolved_config.goal_regions}
    first_terminal_index = None
    first_terminal_phase = None
    for index, phase_label in enumerate(phase_labels):
        if phase_label in terminal_phase_names:
            first_terminal_index = index
            first_terminal_phase = phase_label
            break
    branch_index = _first_index_of_phase(phase_labels, "H1")
    if branch_index is None:
        branch_index = _first_index_of_phase(phase_labels, "L1")
    return TrajectoryPhaseAnnotation(
        phase_labels=phase_labels,
        first_m1_index=_first_index_of_phase(phase_labels, "M1"),
        first_branch_index=branch_index,
        first_m2_index=_first_index_of_phase(phase_labels, "M2"),
        first_terminal_index=first_terminal_index,
        first_terminal_phase=first_terminal_phase,
    )


def _workspace_to_pixel(
    point_xy: tuple[float, float], config: MapConfig, image_size: tuple[int, int]
) -> tuple[int, int]:
    x, y = point_xy
    height, width = image_size
    workspace = config.workspace
    px = int(
        round((x - workspace.xmin) / max(workspace.xmax - workspace.xmin, 1e-08) * (width - 1))
    )
    py = int(
        round((workspace.ymax - y) / max(workspace.ymax - workspace.ymin, 1e-08) * (height - 1))
    )
    return (int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1)))


def _fill_rectangle(
    image: np.ndarray,
    region: RectangleRegion | SemanticRegion,
    color: tuple[int, int, int],
    config: MapConfig,
) -> None:
    if isinstance(region, SemanticRegion):
        for rectangle in region.rectangles:
            _fill_rectangle(image, rectangle, color, config=config)
        return
    top_left = _workspace_to_pixel(
        (region.xmin, region.ymax), config=config, image_size=image.shape[:2]
    )
    bottom_right = _workspace_to_pixel(
        (region.xmax, region.ymin), config=config, image_size=image.shape[:2]
    )
    x0, x1 = sorted((top_left[0], bottom_right[0]))
    y0, y1 = sorted((top_left[1], bottom_right[1]))
    image[y0 : y1 + 1, x0 : x1 + 1] = np.asarray(color, dtype=np.uint8)


def _draw_disk(
    image: np.ndarray, center_px: tuple[int, int], radius: int, color: tuple[int, int, int]
) -> None:
    cx, cy = center_px
    height, width = image.shape[:2]
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 <= radius * radius
    image[mask] = np.asarray(color, dtype=np.uint8)


def make_schematic_base_image(
    config: MapConfig, image_size: int = DEFAULT_VIDEO_IMAGE_SIZE
) -> np.ndarray:
    base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    base_image[:] = np.asarray(SCHEMATIC_BACKGROUND_COLOR, dtype=np.uint8)
    for obstacle in config.obstacle_rectangles:
        _fill_rectangle(base_image, obstacle, SCHEMATIC_OBSTACLE_COLOR, config=config)
    _fill_rectangle(base_image, config.start_room, SCHEMATIC_START_COLOR, config=config)
    _fill_rectangle(base_image, config.merge_region_m1, SCHEMATIC_SHARED_COLOR, config=config)
    _fill_rectangle(base_image, config.middle_shared_region, SCHEMATIC_SHARED_COLOR, config=config)
    _fill_rectangle(
        base_image, config.post_branch_shared_region, SCHEMATIC_SHARED_COLOR, config=config
    )
    _fill_rectangle(base_image, config.merge_region_m2, SCHEMATIC_SHARED_COLOR, config=config)
    _fill_rectangle(base_image, config.upper_hole_region, SCHEMATIC_H1_COLOR, config=config)
    _fill_rectangle(base_image, config.lower_hole_region, SCHEMATIC_L1_COLOR, config=config)
    for goal_region in config.goal_regions:
        _fill_rectangle(
            base_image, goal_region, SCHEMATIC_GOAL_COLOR_BY_NAME[goal_region.name], config=config
        )
    return base_image


def render_schematic_frame(
    base_image: np.ndarray, config: MapConfig, probe_xy: tuple[float, float]
) -> np.ndarray:
    frame = base_image.copy()
    height, width = frame.shape[:2]
    probe_px = _workspace_to_pixel(probe_xy, config=config, image_size=(height, width))
    outer_radius = max(3, int(min(height, width) * 0.03))
    inner_radius = max(2, int(outer_radius * 0.65))
    _draw_disk(frame, probe_px, outer_radius, SCHEMATIC_CURSOR_OUTER)
    _draw_disk(frame, probe_px, inner_radius, SCHEMATIC_CURSOR_INNER)
    return frame


class PandaRouteSemanticEnv:
    def __init__(
        self,
        map_config: MapConfig | None = None,
        rng_seed: int = DEFAULT_RANDOM_SEED,
        enable_randomize: bool = DEFAULT_START_RANDOMIZE,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.map_config = build_default_map_config() if map_config is None else map_config
        self.rng = random.Random(rng_seed)
        self.enable_randomize = bool(enable_randomize)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.image_size = int(image_size)
        self.state: tuple[float, float] | None = None
        self.step_count = 0
        self.done = False
        self.task_spec: TaskSpec | None = None
        self.start_region_name: str | None = None
        self.target_goal_name: str | None = None
        self.trajectory: list[tuple[float, float]] = []
        self.last_info: dict[str, Any] = {}
        self._base_image = make_schematic_base_image(
            self.map_config, image_size=max(self.image_size, 32)
        )

    def sample_task_id(self) -> int:
        return int(self.rng.choice(TASK_ID_VALUES))

    def _resolve_start_state(
        self,
        task_start_region: RectangleRegion,
        enable_randomize: bool,
        explicit_start_state: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        if explicit_start_state is not None:
            x = float(explicit_start_state[0])
            y = float(explicit_start_state[1])
            if not task_start_region.contains_point(x, y):
                raise ValueError(
                    f"Explicit start_state {explicit_start_state} is outside task start region {task_start_region.name}."
                )
            if not is_state_valid(x, y, config=self.map_config):
                raise ValueError(f"Explicit start_state {explicit_start_state} is not valid.")
            return (x, y)
        if enable_randomize:
            return sample_valid_state_in_region(task_start_region, self.rng, config=self.map_config)
        center_state = task_start_region.center
        if not is_state_valid(*center_state, config=self.map_config):
            raise RuntimeError(
                f"The deterministic start-region center is not valid free space. region={task_start_region.name}, center={center_state}"
            )
        return center_state

    def reset(
        self,
        task_id: int | None = None,
        enable_randomize: bool | None = None,
        start_state: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        chosen_task_id = self.sample_task_id() if task_id is None else int(task_id)
        self.task_spec = build_task_spec(chosen_task_id)
        self.start_region_name = self.task_spec.start_region_name
        self.target_goal_name = self.task_spec.target_goal_name
        task_start_region = get_task_start_region(chosen_task_id, self.map_config)
        use_randomized_start = (
            self.enable_randomize if enable_randomize is None else bool(enable_randomize)
        )
        self.state = self._resolve_start_state(
            task_start_region=task_start_region,
            enable_randomize=use_randomized_start,
            explicit_start_state=start_state,
        )
        self.trajectory = [self.state]
        self.step_count = 0
        self.done = False
        self.last_info = {
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.target_goal_name,
            "reached_goal": None,
            "success": False,
            "collision_rejected": False,
            "start_randomized": use_randomized_start,
        }
        return self.state

    def get_policy_state(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() before requesting policy state.")
        return np.asarray([self.state[0], self.state[1]], dtype=np.float32)

    def sync_to_state(
        self, state: tuple[float, float], step_count: int | None = None
    ) -> tuple[float, float]:
        if self.task_spec is None or self.target_goal_name is None:
            raise RuntimeError("Call reset() before syncing state.")
        synced_state = (float(state[0]), float(state[1]))
        self.state = synced_state
        if step_count is not None:
            self.step_count = int(step_count)
        if self.trajectory:
            if step_count is not None:
                self.trajectory = self.trajectory[: max(0, int(step_count))]
            self.trajectory.append(synced_state)
        else:
            self.trajectory = [synced_state]
        reached_goal = get_goal_region_for_state(*synced_state, config=self.map_config)
        success = reached_goal is not None and reached_goal.name == self.target_goal_name
        self.done = reached_goal is not None
        self.last_info = {
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "action": None,
            "proposed_state": synced_state,
            "applied_state": synced_state,
            "proposed_state_valid": is_state_valid(*synced_state, config=self.map_config),
            "collision_rejected": False,
            "reached_goal": None if reached_goal is None else reached_goal.name,
            "success": success,
            "partial_observation": self.get_partial_observation(),
            "full_observation": self.get_full_observation(),
        }
        return synced_state

    def get_partial_observation(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call reset() before requesting observations.")
        return {
            "x": float(self.state[0]),
            "y": float(self.state[1]),
            "position": (float(self.state[0]), float(self.state[1])),
            "step_count": int(self.step_count),
            "phase_name": get_phase_name(*self.state, config=self.map_config),
        }

    def get_full_observation(self) -> dict[str, Any]:
        if self.state is None or self.task_spec is None:
            raise RuntimeError("Call reset() before requesting observations.")
        return {
            **self.get_partial_observation(),
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.task_spec.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.task_spec.target_goal_name,
        }

    def is_state_valid(self, state: tuple[float, float]) -> bool:
        return is_state_valid(*state, config=self.map_config)

    def get_goal_region(self, state: tuple[float, float]) -> GoalRegion | None:
        return get_goal_region_for_state(*state, config=self.map_config)

    def get_phase_name(self, state: tuple[float, float]) -> str:
        return get_phase_name(*state, config=self.map_config)

    def annotate_trajectory_phases(
        self, path: list[tuple[float, float]] | tuple[tuple[float, float], ...]
    ) -> TrajectoryPhaseAnnotation:
        return annotate_trajectory_phases(path, config=self.map_config)

    def step(
        self, action: tuple[float, float]
    ) -> tuple[tuple[float, float], float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")
        if self.task_spec is None or self.target_goal_name is None:
            raise RuntimeError("Call reset() before step().")
        dx = float(action[0])
        dy = float(action[1])
        proposed_state = (float(self.state[0] + dx), float(self.state[1] + dy))
        proposed_valid = is_state_valid(*proposed_state, config=self.map_config)
        collision_rejected = not proposed_valid
        next_state = self.state if collision_rejected else proposed_state
        reached_goal = get_goal_region_for_state(*next_state, config=self.map_config)
        success = reached_goal is not None and reached_goal.name == self.target_goal_name
        self.state = next_state
        self.trajectory.append(next_state)
        self.step_count += 1
        self.done = reached_goal is not None
        reward = self.step_penalty
        if success:
            reward += self.goal_reward
        info = {
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "action": (dx, dy),
            "proposed_state": proposed_state,
            "applied_state": next_state,
            "proposed_state_valid": proposed_valid,
            "collision_rejected": collision_rejected,
            "reached_goal": None if reached_goal is None else reached_goal.name,
            "success": success,
            "partial_observation": self.get_partial_observation(),
            "full_observation": self.get_full_observation(),
        }
        self.last_info = info
        return (next_state, reward, self.done, info)

    def render_frame(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() before render_frame().")
        return render_schematic_frame(
            self._base_image,
            config=self.map_config,
            probe_xy=(float(self.state[0]), float(self.state[1])),
        )

    def render(self) -> np.ndarray:
        return self.render_frame()

    def close(self) -> None:
        return None


def _format_geom_rgba(color: tuple[float, float, float, float]) -> str:
    return " ".join((f"{float(channel):.6f}" for channel in color))


def _rectangle_to_mjcf_geom(
    rectangle: RectangleRegion,
    *,
    height: float,
    rgba: tuple[float, float, float, float],
    geom_type: str = "box",
    mass: float = 0.0,
    contype: int = 1,
    conaffinity: int = 1,
) -> str:
    center_x, center_y = rectangle.center
    size_x = (rectangle.xmax - rectangle.xmin) * 0.5
    size_y = (rectangle.ymax - rectangle.ymin) * 0.5
    return f'<geom name="{rectangle.name}" type="{geom_type}" pos="{center_x:.6f} {center_y:.6f} {height * 0.5:.6f}" size="{size_x:.6f} {size_y:.6f} {height * 0.5:.6f}" rgba="{_format_geom_rgba(rgba)}" mass="{mass:.6f}" contype="{contype}" conaffinity="{conaffinity}"/>'


def _normalize_panda_description_variant(variant: str | None) -> str:
    normalized = DEFAULT_PANDA_DESCRIPTION_VARIANT if variant is None else str(variant).strip()
    if normalized.endswith(".xml"):
        normalized = normalized[:-4]
    if not normalized:
        normalized = DEFAULT_PANDA_DESCRIPTION_VARIANT
    return normalized


def _ensure_local_robot_descriptions_cache() -> None:
    if "ROBOT_DESCRIPTIONS_CACHE" in os.environ:
        return
    if not LOCAL_MUJOCO_MENAGERIE_ROOT.is_dir():
        return
    cache_repo_path = ROBOT_DESCRIPTIONS_CACHE_ROOT / "mujoco_menagerie"
    ROBOT_DESCRIPTIONS_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not cache_repo_path.exists():
        try:
            cache_repo_path.symlink_to(LOCAL_MUJOCO_MENAGERIE_ROOT, target_is_directory=True)
        except OSError:
            shutil.copytree(LOCAL_MUJOCO_MENAGERIE_ROOT, cache_repo_path, dirs_exist_ok=True)
    os.environ["ROBOT_DESCRIPTIONS_CACHE"] = str(ROBOT_DESCRIPTIONS_CACHE_ROOT)


def get_franka_panda_mjcf_path(variant: str | None = DEFAULT_PANDA_DESCRIPTION_VARIANT) -> Path:
    normalized_variant = _normalize_panda_description_variant(variant)
    filename = "panda.xml" if normalized_variant == "panda" else f"{normalized_variant}.xml"
    _ensure_local_robot_descriptions_cache()
    try:
        from robot_descriptions import panda_mj_description
    except Exception:
        panda_package_path = None
    else:
        panda_package_path = Path(panda_mj_description.PACKAGE_PATH)
        candidate_path = panda_package_path / filename
        if candidate_path.is_file():
            return candidate_path.resolve()
    fallback_path = LOCAL_FRANKA_PANDA_ROOT / filename
    if fallback_path.is_file():
        return fallback_path.resolve()
    raise FileNotFoundError(
        f"Could not locate the Franka Emika Panda MJCF description. Tried variant={normalized_variant!r} via robot_descriptions and {fallback_path}."
    )


def _find_named_mjcf_element(root: ET.Element, tag: str, name: str) -> ET.Element | None:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    return None


def _append_route_scene_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if _find_named_mjcf_element(asset, "texture", "ground_tex") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "ground_tex",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.92 0.92 0.90",
                "rgb2": "0.84 0.84 0.82",
                "width": "512",
                "height": "512",
            },
        )
    if _find_named_mjcf_element(asset, "material", "ground_mat") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "ground_mat",
                "texture": "ground_tex",
                "texrepeat": "8 8",
                "specular": "0.0",
                "shininess": "0.0",
                "reflectance": "0.0",
            },
        )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {
                "ambient": "0.35 0.35 0.35",
                "diffuse": "0.85 0.85 0.85",
                "specular": "0.15 0.15 0.15",
            },
        )
    if visual.find("map") is None:
        ET.SubElement(visual, "map", {"znear": "0.001"})


def _append_route_probe_site(root: ET.Element) -> None:
    if _find_named_mjcf_element(root, "site", "probe_tip") is not None:
        return
    attachment_body = _find_named_mjcf_element(root, "body", "attachment")
    if attachment_body is not None:
        ET.SubElement(
            attachment_body,
            "site",
            {
                "name": "probe_tip",
                "type": "sphere",
                "pos": "0 0 0",
                "size": "0.010",
                "rgba": "0.90 0.15 0.15 1.0",
            },
        )
        return
    hand_body = _find_named_mjcf_element(root, "body", "hand")
    if hand_body is not None:
        ET.SubElement(
            hand_body,
            "site",
            {
                "name": "probe_tip",
                "type": "sphere",
                "pos": "0 0 0.103",
                "size": "0.010",
                "rgba": "0.90 0.15 0.15 1.0",
            },
        )
        return
    raise RuntimeError("Could not attach a probe_tip site to the Franka Panda MJCF.")


def _append_route_world_geoms(root: ET.Element, config: MapConfig) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Franka Panda MJCF does not define a worldbody.")
    if _find_named_mjcf_element(worldbody, "geom", "ground") is None:
        worldbody.insert(
            0,
            ET.Element(
                "geom",
                {
                    "name": "ground",
                    "type": "plane",
                    "size": "2 2 0.05",
                    "material": "ground_mat",
                    "contype": "1",
                    "conaffinity": "1",
                },
            ),
        )
    wall_height = 0.035
    marker_height = 0.002
    geoms = [
        _rectangle_to_mjcf_geom(
            config.task_start_regions[0],
            height=marker_height,
            rgba=(0.3, 0.47, 0.66, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.task_start_regions[1],
            height=marker_height,
            rgba=(0.3, 0.47, 0.66, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.merge_region_m1.rectangles[0],
            height=marker_height,
            rgba=(0.55, 0.55, 0.55, 0.45),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.upper_hole_region.rectangles[0],
            height=marker_height,
            rgba=(0.17, 0.62, 0.17, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.lower_hole_region.rectangles[0],
            height=marker_height,
            rgba=(0.58, 0.4, 0.74, 0.55),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.merge_region_m2.rectangles[0],
            height=marker_height,
            rgba=(0.55, 0.55, 0.55, 0.45),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.goal_regions[0],
            height=marker_height,
            rgba=(0.1, 0.62, 0.47, 0.6),
            contype=0,
            conaffinity=0,
        ),
        _rectangle_to_mjcf_geom(
            config.goal_regions[1],
            height=marker_height,
            rgba=(0.85, 0.37, 0.08, 0.6),
            contype=0,
            conaffinity=0,
        ),
    ]
    geoms.extend(
        (
            _rectangle_to_mjcf_geom(obstacle, height=wall_height, rgba=(0.28, 0.28, 0.3, 1.0))
            for obstacle in config.obstacle_rectangles
        )
    )
    for geom_xml in geoms:
        geom = ET.fromstring(geom_xml)
        if _find_named_mjcf_element(worldbody, "geom", geom.get("name", "")) is None:
            worldbody.append(geom)


def build_panda_route_mjcf(
    config: MapConfig, robot_variant: str | None = DEFAULT_PANDA_DESCRIPTION_VARIANT
) -> str:
    base_xml_path = get_franka_panda_mjcf_path(robot_variant)
    root = ET.fromstring(base_xml_path.read_text())
    root.set("model", "panda_route")
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((base_xml_path.parent / "assets").resolve()))
    _append_route_scene_assets(root)
    _append_route_probe_site(root)
    _append_route_world_geoms(root, config)
    return ET.tostring(root, encoding="unicode")


def _resolve_arm_joint_names(model: "mujoco.MjModel") -> tuple[str, ...]:
    for joint_names in (FRANKA_ARM_JOINT_NAMES, LEGACY_ARM_JOINT_NAMES):
        joint_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            for joint_name in joint_names
        ]
        if all((joint_id >= 0 for joint_id in joint_ids)):
            return joint_names
    raise RuntimeError("Could not resolve the Panda arm joints in the MuJoCo model.")


def _resolve_optional_qpos_indices(
    model: "mujoco.MjModel", joint_names: tuple[str, ...]
) -> np.ndarray:
    indices: list[int] = []
    for joint_name in joint_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
        if joint_id < 0:
            continue
        indices.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(indices, dtype=np.int64)


def _default_home_qpos_for_joint_names(joint_names: tuple[str, ...]) -> np.ndarray:
    if joint_names == FRANKA_ARM_JOINT_NAMES:
        return FRANKA_HOME_QPOS.copy()
    return LEGACY_HOME_QPOS.copy()


def _resolve_probe_site_name(model: "mujoco.MjModel") -> str:
    for site_name in ("probe_tip", "attachment_site"):
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
        if site_id >= 0:
            return site_name
    raise RuntimeError("Could not resolve a probe site in the MuJoCo model.")


class PandaRouteMjEnv(PandaRouteSemanticEnv):
    PROBE_TARGET_Z = 0.1
    IK_TOLERANCE = 0.0005
    IK_ACCEPT_RESIDUAL = 0.01
    IK_MAX_ITERS = 240
    IK_DAMPING = 0.02
    IK_REGULARIZATION = 0.01
    RENDER_LOOKAT = np.asarray([0.45, 0.0, 0.24], dtype=np.float64)
    RENDER_DISTANCE = 1.35
    RENDER_AZIMUTH = 135.0
    RENDER_ELEVATION = -28.0

    def __init__(
        self,
        map_config: MapConfig | None = None,
        rng_seed: int = DEFAULT_RANDOM_SEED,
        enable_randomize: bool = DEFAULT_START_RANDOMIZE,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        image_size: int = DEFAULT_IMAGE_SIZE,
        robot_variant: str = DEFAULT_PANDA_DESCRIPTION_VARIANT,
    ) -> None:
        if mujoco is None:
            raise RuntimeError(
                "mujoco is required for PandaRouteMjEnv. Install it first."
            ) from _MUJOCO_IMPORT_ERROR
        super().__init__(
            map_config=map_config,
            rng_seed=rng_seed,
            enable_randomize=enable_randomize,
            step_penalty=step_penalty,
            goal_reward=goal_reward,
            image_size=image_size,
        )
        self.robot_variant = _normalize_panda_description_variant(robot_variant)
        self.model = mujoco.MjModel.from_xml_string(
            build_panda_route_mjcf(self.map_config, robot_variant=self.robot_variant)
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = self.RENDER_LOOKAT
        self.camera.distance = float(self.RENDER_DISTANCE)
        self.camera.azimuth = float(self.RENDER_AZIMUTH)
        self.camera.elevation = float(self.RENDER_ELEVATION)
        self.joint_names = _resolve_arm_joint_names(self.model)
        self.joint_ids = tuple(
            (
                int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
                for joint_name in self.joint_names
            )
        )
        if any((joint_id < 0 for joint_id in self.joint_ids)):
            raise RuntimeError("Could not resolve Panda joint ids in MuJoCo model.")
        self.qpos_indices = np.asarray(
            [int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids], dtype=np.int64
        )
        self.dof_indices = np.asarray(
            [int(self.model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids], dtype=np.int64
        )
        self.joint_range = self.model.jnt_range[np.asarray(self.joint_ids, dtype=np.intp)].astype(
            np.float64
        )
        self.gripper_qpos_indices = _resolve_optional_qpos_indices(
            self.model, FRANKA_FINGER_JOINT_NAMES
        )
        self.home_keyframe_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        self.site_name = _resolve_probe_site_name(self.model)
        self.site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name))
        if self.site_id < 0:
            raise RuntimeError("Could not resolve a probe site in the MuJoCo model.")
        self.home_qpos = _default_home_qpos_for_joint_names(self.joint_names)
        self._reset_robot_pose()
        self.home_qpos = self.data.qpos[self.qpos_indices].astype(np.float64, copy=True)
        self.action_dim = int(self.qpos_indices.size)

    def _reset_robot_pose(self) -> None:
        if self.home_keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_keyframe_id)
        else:
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.qpos[self.qpos_indices] = self.home_qpos
            if self.gripper_qpos_indices.size > 0:
                self.data.qpos[self.gripper_qpos_indices] = FRANKA_OPEN_GRIPPER_QPOS
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].astype(np.float32, copy=True)

    def get_probe_state(self) -> tuple[float, float]:
        probe_position = self.get_probe_position_3d()
        return (float(probe_position[0]), float(probe_position[1]))

    def get_probe_position_3d(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].astype(np.float32, copy=True)

    def get_full_observation(self) -> dict[str, Any]:
        base = super().get_full_observation()
        base.update(
            {
                "joint_positions": self.get_joint_positions().tolist(),
                "probe_position_3d": self.get_probe_position_3d().tolist(),
            }
        )
        return base

    def _augment_info_with_robot(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            **info,
            "joint_positions": self.get_joint_positions().tolist(),
            "probe_position_3d": self.get_probe_position_3d().tolist(),
            "partial_observation": self.get_partial_observation(),
            "full_observation": self.get_full_observation(),
        }

    def _clip_joint_positions(
        self, joint_positions: np.ndarray | list[float] | tuple[float, ...]
    ) -> np.ndarray:
        joint_array = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if int(joint_array.shape[0]) != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} joint positions, got shape={tuple(joint_array.shape)}."
            )
        return np.clip(joint_array, self.joint_range[:, 0], self.joint_range[:, 1])

    def _set_joint_positions(
        self, joint_positions: np.ndarray | list[float] | tuple[float, ...]
    ) -> np.ndarray:
        clipped = self._clip_joint_positions(joint_positions)
        self.data.qpos[self.qpos_indices] = clipped
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return clipped.astype(np.float32, copy=False)

    def _solve_inverse_kinematics(
        self, target_xyz: np.ndarray, q_init: np.ndarray | None = None
    ) -> np.ndarray:
        q = self.home_qpos.copy() if q_init is None else np.asarray(q_init, dtype=np.float64).copy()
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)
        for _ in range(self.IK_MAX_ITERS):
            self.data.qpos[self.qpos_indices] = q
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            current_xyz = self.data.site_xpos[self.site_id]
            error = np.asarray(target_xyz - current_xyz, dtype=np.float64)
            if float(np.linalg.norm(error)) <= self.IK_TOLERANCE:
                return q
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = jacp[:, self.dof_indices]
            system = jac @ jac.T + self.IK_DAMPING**2 * eye
            dq = jac.T @ np.linalg.solve(system, error)
            dq += self.IK_REGULARIZATION * (self.home_qpos - q)
            q = q + dq
            q = np.clip(q, self.joint_range[:, 0], self.joint_range[:, 1])
        self.data.qpos[self.qpos_indices] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        residual = float(np.linalg.norm(self.data.site_xpos[self.site_id] - target_xyz))
        if residual > self.IK_ACCEPT_RESIDUAL:
            raise RuntimeError(
                f"IK did not converge for target_xyz={target_xyz.tolist()}, residual={residual:.6f}"
            )
        return q

    def _sync_robot_to_probe_state(self, probe_state: tuple[float, float]) -> None:
        target_xyz = np.asarray(
            [float(probe_state[0]), float(probe_state[1]), self.PROBE_TARGET_Z], dtype=np.float64
        )
        q_now = self.data.qpos[self.qpos_indices].astype(np.float64, copy=True)
        solved_q = self._solve_inverse_kinematics(target_xyz=target_xyz, q_init=q_now)
        self.data.qpos[self.qpos_indices] = solved_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        task_id: int | None = None,
        enable_randomize: bool | None = None,
        start_state: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        state = super().reset(
            task_id=task_id, enable_randomize=enable_randomize, start_state=start_state
        )
        self._reset_robot_pose()
        self._sync_robot_to_probe_state(state)
        self.last_info = self._augment_info_with_robot(self.last_info)
        return state

    def step(
        self, action: np.ndarray | list[float] | tuple[float, ...]
    ) -> tuple[tuple[float, float], float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")
        if self.task_spec is None or self.target_goal_name is None:
            raise RuntimeError("Call reset() before step().")
        previous_state = (float(self.state[0]), float(self.state[1]))
        previous_joint_positions = self.get_joint_positions()
        target_joint_positions = self._set_joint_positions(action)
        proposed_state = self.get_probe_state()
        proposed_valid = is_state_valid(*proposed_state, config=self.map_config)
        collision_rejected = not proposed_valid
        if collision_rejected:
            self._set_joint_positions(previous_joint_positions)
            next_state = previous_state
        else:
            next_state = proposed_state
        reached_goal = get_goal_region_for_state(*next_state, config=self.map_config)
        success = reached_goal is not None and reached_goal.name == self.target_goal_name
        self.state = next_state
        self.trajectory.append(next_state)
        self.step_count += 1
        self.done = reached_goal is not None
        reward = self.step_penalty
        if success:
            reward += self.goal_reward
        info = {
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "start_region_name": self.start_region_name,
            "branch_region_name": self.task_spec.branch_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "action": target_joint_positions.astype(np.float32).tolist(),
            "proposed_state": proposed_state,
            "applied_state": next_state,
            "proposed_state_valid": proposed_valid,
            "collision_rejected": collision_rejected,
            "reached_goal": None if reached_goal is None else reached_goal.name,
            "success": success,
        }
        info = self._augment_info_with_robot(info)
        self.last_info = info
        return (next_state, reward, self.done, info)

    def sync_to_state(
        self, state: tuple[float, float], step_count: int | None = None
    ) -> tuple[float, float]:
        self._sync_robot_to_probe_state((float(state[0]), float(state[1])))
        synced_state = super().sync_to_state(state=state, step_count=step_count)
        self.last_info = self._augment_info_with_robot(self.last_info)
        return synced_state

    def sync_to_joint_positions(
        self,
        joint_positions: np.ndarray | list[float] | tuple[float, ...],
        *,
        probe_state: tuple[float, float] | None = None,
        step_count: int | None = None,
    ) -> tuple[float, float]:
        self._set_joint_positions(joint_positions)
        resolved_probe_state = (
            self.get_probe_state()
            if probe_state is None
            else (float(probe_state[0]), float(probe_state[1]))
        )
        synced_state = super().sync_to_state(state=resolved_probe_state, step_count=step_count)
        self.last_info = self._augment_info_with_robot(self.last_info)
        return synced_state

    def render_frame(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() before render_frame().")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model, height=self.image_size, width=self.image_size
            )
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()


def start_ffmpeg_raw_writer(output_path: Path, width: int, height: int, fps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to export LeRobot videos.") from exc


def get_eval_defaults(policy_type: str) -> dict[str, Any]:
    return load_policy_mode_defaults("eval", ENV_NAME, policy_type)


def _compute_online_signature_prefix(
    state_history: deque[np.ndarray], sig_depth: int, signature_backend: str
) -> np.ndarray:
    if len(state_history) == 0:
        raise ValueError("state_history is empty; cannot compute path signature.")
    window = np.stack(list(state_history), axis=0).astype(np.float32, copy=False)
    return compute_signatory_signature_np(window, sig_depth)
    return compute_signatory_signature_np(window, sig_depth)


def build_balanced_task_schedule(num_rollouts: int, seed: int) -> list[int]:
    repeated = [TASK_ID_VALUES[index % len(TASK_ID_VALUES)] for index in range(num_rollouts)]
    rng = np.random.default_rng(seed)
    rng.shuffle(repeated)
    return [int(task_id) for task_id in repeated]


def detect_route_mismatch(task_spec: TaskSpec, phase_name: str) -> dict[str, str] | None:
    if phase_name in {"H1", "L1"} and phase_name != task_spec.branch_region_name:
        return {
            "stage": "branch",
            "expected_phase": task_spec.branch_region_name,
            "observed_phase": phase_name,
        }
    if phase_name in {"G1", "G2"} and phase_name != task_spec.target_goal_name:
        return {
            "stage": "goal",
            "expected_phase": task_spec.target_goal_name,
            "observed_phase": phase_name,
        }
    return None


def evaluate_policy(
    *, policy_type: str, args, policy, cfg, preprocessor, postprocessor, policy_dir: Path
) -> None:
    import torch

    eval_num_rollouts = int(getattr(args, "eval_num_rollouts", args.num_rollouts))
    eval_max_steps = int(getattr(args, "eval_max_steps", args.max_steps))
    eval_fps = int(getattr(args, "eval_fps", args.fps))
    eval_seed = int(getattr(args, "eval_seed", args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_features = (
        cfg.visual_observation_features
        if hasattr(cfg, "visual_observation_features")
        else cfg.image_features
    )
    if len(visual_features) == 0:
        raise RuntimeError("Policy has no image input feature; visual eval assumes image input.")
    image_key = list(visual_features.keys())[0]
    image_shape = tuple(visual_features[image_key].shape)
    image_hw = (int(image_shape[1]), int(image_shape[2]))
    if image_hw[0] != image_hw[1]:
        raise RuntimeError("panda_route evaluation expects square image inputs.")
    if cfg.robot_state_feature is None:
        raise RuntimeError("Policy has no observation.state feature.")
    state_key = "observation.state"
    state_dim = int(cfg.robot_state_feature.shape[0])
    resolve_policy_capability_flags(cfg)
    signature_key = DEFAULT_PATH_SIGNATURE_KEY
    signature_backend = None
    signature_backend = resolve_signature_backend("signatory")
    if int(cfg.signature_depth) <= 0:
        raise ValueError(f"Invalid cfg.signature_depth={cfg.signature_depth}. Must be > 0.")
    if int(cfg.signature_dim) <= 0:
        raise ValueError(
            f"Invalid cfg.signature_dim={cfg.signature_dim}. Expected positive signature dimension when use_path_signature=True."
        )
    print(
        f"[info] online path-signature enabled: backend={signature_backend}, history=full_prefix, depth={cfg.signature_depth}, dim={cfg.signature_dim}"
    )
    print(
        f"[info] online delta-signature enabled: key={DEFAULT_DELTA_SIGNATURE_KEY}, rule=g_t-g_(t-1), first_step=zeros"
    )
    initial_memory_debug = get_visual_memory_debug_stats(policy)
    print(
        "[info] visual prefix memory online update enabled: "
        + "rollout uses signature-indexed slot memory without rebuilding explicit prefix-sequence tensors each step"
    )
    if initial_memory_debug is not None:
        print(
            f"[info] visual prefix memory debug: enabled={bool(initial_memory_debug.get('enabled', False))}, num_slots={int(initial_memory_debug.get('num_slots', 0))}, updates={int(initial_memory_debug.get('update_count', 0))}"
        )
    map_config = build_default_map_config()
    enable_randomize = bool(getattr(args, "enable_randomize", False))
    print(
        "[info] panda_route eval reset mode: "
        + ("randomized starts" if enable_randomize else "deterministic center starts")
    )
    env = PandaRouteMjEnv(
        map_config=map_config,
        rng_seed=eval_seed,
        enable_randomize=enable_randomize,
        image_size=image_hw[0],
    )
    if state_dim != env.action_dim:
        raise RuntimeError(
            f"Loaded panda_route policy uses an incompatible observation.state dim. Policy expects {state_dim}, but joint-space panda_route requires {env.action_dim}. Please retrain on the regenerated joint-space dataset."
        )
    action_feature = getattr(cfg, "action_feature", None)
    if action_feature is not None and hasattr(action_feature, "shape"):
        action_dim = int(action_feature.shape[0])
        if action_dim != env.action_dim:
            raise RuntimeError(
                f"Loaded panda_route policy uses an incompatible action dim. Policy expects {action_dim}, but joint-space panda_route requires {env.action_dim}. Please retrain on the regenerated joint-space dataset."
            )
    print(
        f"[info] panda_route eval action semantics: observation.state=joint_positions, action=joint_position_target, per_joint_delta_clip={args.max_action_step:.4f}"
    )
    results = []
    success_count = 0
    route_failure_count = 0
    task_success_counts = {int(task_id): 0 for task_id in TASK_ID_VALUES}
    task_rollout_counts = {int(task_id): 0 for task_id in TASK_ID_VALUES}
    task_route_failure_counts = {int(task_id): 0 for task_id in TASK_ID_VALUES}
    task_schedule = build_balanced_task_schedule(eval_num_rollouts, eval_seed)
    try:
        for ep_idx, task_id in enumerate(task_schedule):
            task_spec = build_task_spec(task_id)
            task_rollout_counts[task_id] += 1
            state_xy = tuple((float(v) for v in env.reset(task_id=task_id)))
            joint_state = env.get_joint_positions()
            if env.start_region_name != task_spec.start_region_name:
                raise RuntimeError(
                    f"Environment reset returned a start region that does not match the task. task={task_spec.task_code}, expected_start={task_spec.start_region_name}, got={env.start_region_name}"
                )
            if hasattr(policy, "reset"):
                policy.reset()
            episode_reward = 0.0
            video_path = output_dir / f"rollout_{ep_idx:03d}_task_{task_spec.task_code}.mp4"
            writer = start_ffmpeg_raw_writer(video_path, image_hw[1], image_hw[0], eval_fps)
            if writer.stdin is None:
                raise RuntimeError("Failed to open ffmpeg stdin for rollout video writing.")
            trajectory = [state_xy]
            joint_history = deque()
            previous_signature_vec = None
            last_info = {
                **env.last_info,
                "phase_name": env.get_phase_name(state_xy),
                "route_mismatch": False,
                "route_mismatch_stage": None,
                "expected_phase": None,
                "observed_phase": None,
                "failure_reason": None,
            }
            for _step_idx in range(eval_max_steps):
                frame = env.render_frame()
                writer.stdin.write(frame.astype(np.uint8).tobytes())
                obs = build_eval_observation(
                    state_xy=state_xy,
                    rgb_frame=frame,
                    state_key=state_key,
                    image_key=image_key,
                    state_dim=state_dim,
                    state_vector=joint_state,
                )
                assert joint_history is not None
                joint_history.append(joint_state.astype(np.float32, copy=True))
                signature_vec = _compute_online_signature_prefix(
                    state_history=joint_history,
                    sig_depth=int(cfg.signature_depth),
                    signature_backend=str(signature_backend),
                )
                if signature_vec.shape[0] != int(cfg.signature_dim):
                    raise RuntimeError(
                        f"Online signature dimension mismatch: got {signature_vec.shape[0]}, expected cfg.signature_dim={cfg.signature_dim}."
                    )
                obs[signature_key] = torch.from_numpy(signature_vec.astype(np.float32, copy=False))
                delta_signature_vec = compute_delta_signature_step_np(
                    signature_vec, previous_signature_vec
                )
                obs[DEFAULT_DELTA_SIGNATURE_KEY] = torch.from_numpy(
                    delta_signature_vec.astype(np.float32, copy=False)
                )
                previous_signature_vec = signature_vec.astype(np.float32, copy=True)
                obs = preprocessor(obs)
                if signature_key not in obs:
                    raise KeyError(
                        f"`{signature_key}` missing after preprocessor; cannot run policy with use_path_signature=True."
                    )
                path_signature = obs[signature_key]
                if path_signature.ndim == 1:
                    path_signature = path_signature.unsqueeze(0)
                elif path_signature.ndim != 2:
                    raise RuntimeError(
                        f"`{signature_key}` must be 1D/2D after preprocessing, got shape={tuple(path_signature.shape)}"
                    )
                obs[signature_key] = path_signature.to(
                    device=obs[state_key].device, dtype=obs[state_key].dtype
                )
                if DEFAULT_DELTA_SIGNATURE_KEY not in obs:
                    raise KeyError(
                        f"`{DEFAULT_DELTA_SIGNATURE_KEY}` missing after preprocessor; cannot run policy with use_delta_signature=True."
                    )
                delta_signature = obs[DEFAULT_DELTA_SIGNATURE_KEY]
                if delta_signature.ndim == 1:
                    delta_signature = delta_signature.unsqueeze(0)
                elif delta_signature.ndim != 2:
                    raise RuntimeError(
                        f"`{DEFAULT_DELTA_SIGNATURE_KEY}` must be 1D/2D after preprocessing, got shape={tuple(delta_signature.shape)}"
                    )
                obs[DEFAULT_DELTA_SIGNATURE_KEY] = delta_signature.to(
                    device=obs[state_key].device, dtype=obs[state_key].dtype
                )
                with torch.no_grad():
                    action = policy.select_action(obs)
                action = postprocessor(action)
                action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                if action_np.ndim != 1 or int(action_np.shape[0]) < env.action_dim:
                    raise RuntimeError(
                        f"panda_route evaluation expects a 1D joint-position action with at least {env.action_dim} values, got shape={tuple(action_np.shape)}."
                    )
                target_joint = action_np[: env.action_dim].copy()
                if args.max_action_step > 0.0:
                    joint_delta = target_joint - joint_state
                    joint_delta = np.clip(
                        joint_delta, -float(args.max_action_step), float(args.max_action_step)
                    )
                    target_joint = joint_state + joint_delta
                next_state, reward, done, info = env.step(target_joint)
                state_xy = (float(next_state[0]), float(next_state[1]))
                joint_state = env.get_joint_positions()
                trajectory.append(state_xy)
                episode_reward += float(reward)
                phase_name = env.get_phase_name(state_xy)
                mismatch = detect_route_mismatch(task_spec=task_spec, phase_name=phase_name)
                last_info = {
                    **info,
                    "phase_name": phase_name,
                    "route_mismatch": False,
                    "route_mismatch_stage": None,
                    "expected_phase": None,
                    "observed_phase": None,
                    "failure_reason": None,
                }
                if mismatch is not None:
                    last_info = {
                        **last_info,
                        "route_mismatch": True,
                        "route_mismatch_stage": mismatch["stage"],
                        "expected_phase": mismatch["expected_phase"],
                        "observed_phase": mismatch["observed_phase"],
                        "failure_reason": "wrong_route",
                        "success": False,
                    }
                    final_frame = env.render_frame()
                    writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                    break
                if done:
                    final_frame = env.render_frame()
                    writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                    break
            writer.stdin.close()
            code = writer.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed on rollout {ep_idx} with exit code {code}")
            success = bool(last_info.get("success", False)) and (
                not bool(last_info.get("route_mismatch", False))
            )
            if success:
                success_count += 1
                task_success_counts[task_id] += 1
            if last_info.get("route_mismatch", False):
                route_failure_count += 1
                task_route_failure_counts[task_id] += 1
            result = {
                "episode_index": ep_idx,
                "task_id": task_id,
                "task_code": task_spec.task_code,
                "start_region_name": str(last_info.get("start_region_name")),
                "target_goal_name": str(last_info.get("target_goal_name")),
                "video_path": str(video_path),
                "final_position": [float(state_xy[0]), float(state_xy[1])],
                "final_joint_positions": joint_state.astype(np.float32).tolist(),
                "final_phase_name": str(last_info.get("phase_name")),
                "reached_goal": last_info.get("reached_goal"),
                "route_mismatch": bool(last_info.get("route_mismatch", False)),
                "route_mismatch_stage": last_info.get("route_mismatch_stage"),
                "expected_phase": last_info.get("expected_phase"),
                "observed_phase": last_info.get("observed_phase"),
                "failure_reason": last_info.get("failure_reason"),
                "success": success,
                "steps": int(len(trajectory) - 1),
                "sum_reward": float(episode_reward),
                "collision_rejections": int(
                    sum(
                        (
                            1
                            for index in range(1, len(env.trajectory))
                            if env.trajectory[index] == env.trajectory[index - 1]
                        )
                    )
                ),
            }
            memory_debug_stats = get_visual_memory_debug_stats(policy)
            if memory_debug_stats is not None:
                result["visual_memory_debug"] = memory_debug_stats
            results.append(result)
            memory_debug_text = ""
            if memory_debug_stats is not None:
                memory_debug_text = f" memory_updates={int(memory_debug_stats.get('update_count', 0))} memory_norm={float(memory_debug_stats.get('state_norm', 0.0)):.4f}"
            print(
                f"[{ep_idx + 1:03d}/{eval_num_rollouts:03d}] task={task_spec.task_code}->{task_spec.target_goal_name} success={success} steps={result['steps']} reached={result['reached_goal']} route_mismatch={result['route_mismatch']} video={video_path.name}{memory_debug_text}"
            )
    finally:
        env.close()
    per_task = {
        str(task_id): {
            "goal_name": TASK_ID_TO_GOAL_NAME[task_id],
            "rollouts": int(task_rollout_counts[task_id]),
            "success_count": int(task_success_counts[task_id]),
            "wrong_route_failures": int(task_route_failure_counts[task_id]),
            "success_rate": float(
                task_success_counts[task_id] / max(1, task_rollout_counts[task_id])
            ),
        }
        for task_id in TASK_ID_VALUES
    }
    summary = {
        "env": ENV_NAME,
        "policy_type": policy_type,
        "num_rollouts": eval_num_rollouts,
        "success_count": success_count,
        "success_rate": float(success_count / max(1, eval_num_rollouts)),
        "wrong_route_failures": route_failure_count,
        "seed": eval_seed,
        "fps": eval_fps,
        "max_steps": eval_max_steps,
        "max_action_step": args.max_action_step,
        "start_randomized": enable_randomize,
        "policy_dir": str(policy_dir),
        "per_task": per_task,
        "results": results,
    }
    memory_debug_stats = get_visual_memory_debug_stats(policy)
    if memory_debug_stats is not None:
        summary["visual_memory_debug"] = memory_debug_stats
    summary_path = write_summary(output_dir, summary)
    print(f"\nSaved {eval_num_rollouts} rollout videos to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Success rate: {summary['success_rate']:.3f} ({success_count}/{eval_num_rollouts})")
    print(f"Wrong-route failures: {route_failure_count}/{eval_num_rollouts}")
    for task_id in TASK_ID_VALUES:
        task_summary = per_task[str(task_id)]
        print(
            f"  task {task_id} ({task_summary['goal_name']}): {task_summary['success_count']}/{task_summary['rollouts']} = {task_summary['success_rate']:.3f}, wrong_route={task_summary['wrong_route_failures']}"
        )
