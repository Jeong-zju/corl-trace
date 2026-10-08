from __future__ import annotations
from collections import deque
import math
import random
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
except Exception as exc:
    plt = None
    Line2D = None
    Patch = None
    Rectangle = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None
from eval_helpers import (
    build_eval_observation,
    compute_delta_signature_step_np,
    compute_signatory_signature_np,
    resolve_signature_backend,
    resolve_single_visual_observation_feature,
    write_summary,
)
from policy_capabilities import get_visual_memory_debug_stats, resolve_policy_capability_flags
from policy_defaults import load_policy_mode_defaults

DEFAULT_RANDOM_SEED = 7
DEFAULT_MAX_SAMPLE_ATTEMPTS = 10000
DEFAULT_DT = 1.0
DEFAULT_STEP_PENALTY = -0.01
DEFAULT_GOAL_REWARD = 1.0
DEFAULT_RANDOM_ROLLOUT_STEPS = 120
DEFAULT_RANDOM_ACTION_SCALE = 2.5
DEFAULT_RENDER_EVERY_N_STEPS = 5
DEFAULT_RENDER_PAUSE_SECONDS = 0.05
DEFAULT_MANUAL_TEST_RESETS = 4
DEFAULT_MANUAL_TEST_STEPS = 7
VALID_SAMPLE_COLOR = "#1f77b4"
ROBOT_MARKER_SIZE = 90
TRAJECTORY_LINEWIDTH = 2.0
TASK_ID_TO_GOAL_NAME = {0: "G00", 1: "G01", 2: "G10", 3: "G11"}
PHASE_COLOR_BY_NAME = {
    "start_region": "#4c78a8",
    "shared_corridor_region": "#72b7b2",
    "decision_region_H1": "#c0392b",
    "branch1_upper_region": "#2ca02c",
    "branch1_lower_region": "#9467bd",
    "merge_region_1": "#7f7f7f",
    "middle_corridor_region": "#5ab4ac",
    "decision_region_H2": "#ff7f0e",
    "branch2_upper_region": "#1b9e77",
    "branch2_lower_region": "#d95f02",
    "merge_region_2": "#8c8c8c",
    "final_corridor_region": "#80b1d3",
    "G00": "#1b9e77",
    "G01": "#66a61e",
    "G10": "#d95f02",
    "G11": "#e6ab02",
    "free_space_other": "#9ecae1",
    "obstacle": "#7f7f7f",
    "out_of_bounds": "#111111",
}
ENV_NAME = "braidedhub"
DEFAULT_COLLISION_MODE = "reject"
BRANCH1_PHASE_BY_BIT = {0: "branch1_upper_region", 1: "branch1_lower_region"}
BRANCH2_PHASE_BY_BIT = {0: "branch2_upper_region", 1: "branch2_lower_region"}
DEFAULT_PATH_SIGNATURE_KEY = "observation.path_signature"
DEFAULT_DELTA_SIGNATURE_KEY = "observation.delta_signature"
DEFAULT_VIDEO_IMAGE_SIZE = 128
MAP_BACKGROUND_COLOR = (248, 244, 227)
OBSTACLE_COLOR = (74, 74, 74)
START_COLOR = (76, 120, 168)
GOAL_COLOR_BY_NAME = {
    "G00": (27, 158, 119),
    "G01": (102, 166, 30),
    "G10": (217, 95, 2),
    "G11": (230, 171, 2),
}
ROBOT_OUTER_COLOR = (255, 255, 255)
ROBOT_INNER_COLOR = (240, 70, 70)
VALID_COLLISION_MODES = ("reject",)


@dataclass(frozen=True)
class RectangleRegion:
    """Axis-aligned rectangle in world coordinates.

    Boundary convention:
    - Rectangle membership is closed: points on the rectangle boundary count
      as inside the rectangle.
    """

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
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def as_patch(self, **kwargs) -> Rectangle:
        if Rectangle is None:
            raise RuntimeError(
                "matplotlib is required for plotting. Install it first, for example: pip install matplotlib"
            ) from _MATPLOTLIB_IMPORT_ERROR
        return Rectangle((self.xmin, self.ymin), self.width, self.height, **kwargs)


@dataclass(frozen=True)
class RectangleObstacle(RectangleRegion):
    """Obstacle rectangle used by planners or collision checkers later."""


@dataclass(frozen=True)
class GoalRegion(RectangleRegion):
    """Terminal region on the right side of the map."""


@dataclass(frozen=True)
class DecisionPoint:
    """Named navigation decision point."""

    name: str
    x: float
    y: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class MapConfig:
    """Container for all geometry needed by the current static map."""

    workspace: RectangleRegion
    start_region: "SemanticRegion"
    task_start_regions: tuple[RectangleRegion, ...]
    shared_corridor_region: "SemanticRegion"
    decision_region_h1: "SemanticRegion"
    branch1_upper_region: "SemanticRegion"
    branch1_lower_region: "SemanticRegion"
    branch1_upper_detection_region: RectangleRegion
    branch1_lower_detection_region: RectangleRegion
    merge_region_1: "SemanticRegion"
    middle_corridor_region: "SemanticRegion"
    decision_region_h2: "SemanticRegion"
    branch2_upper_region: "SemanticRegion"
    branch2_lower_region: "SemanticRegion"
    branch2_upper_detection_region: RectangleRegion
    branch2_lower_detection_region: RectangleRegion
    merge_region_2: "SemanticRegion"
    final_corridor_region: "SemanticRegion"
    free_space_rectangles: tuple[RectangleRegion, ...]
    obstacle_rectangles: tuple[RectangleObstacle, ...]
    goal_regions: tuple[GoalRegion, ...]
    terminal_regions: tuple[GoalRegion, ...]
    decision_points: tuple[DecisionPoint, ...]
    path_labels: tuple[tuple[str, tuple[float, float]], ...]


@dataclass(frozen=True)
class TaskSpec:
    """Task definition for the 4-goal 2-bit implicit-cue setting."""

    task_id: int
    task_code: str
    task_bits: tuple[int, int]
    start_region_name: str
    target_goal_name: str


@dataclass(frozen=True)
class SemanticRegion:
    """Named semantic region approximated by a union of rectangles."""

    name: str
    rectangles: tuple[RectangleRegion, ...]

    def contains_point(self, x: float, y: float) -> bool:
        return any((rectangle.contains_point(x, y) for rectangle in self.rectangles))


@dataclass(frozen=True)
class TrajectoryPhaseAnnotation:
    """Phase labels and first-entry events for a 2D trajectory."""

    phase_labels: tuple[str, ...]
    first_h1_index: int | None
    first_h2_index: int | None
    first_terminal_index: int | None
    first_terminal_phase: str | None


class SupportsUniform(Protocol):
    """Minimal RNG protocol for workspace and free-space sampling."""

    def uniform(self, a: float, b: float) -> float: ...


def build_obstacles_from_free_rectangles(
    workspace: RectangleRegion, free_rectangles: tuple[RectangleRegion, ...]
) -> tuple[RectangleObstacle, ...]:
    """Convert workspace minus free rectangles into merged obstacle rectangles.

    The decomposition uses only the rectangle boundary coordinates already
    present in the map config. This keeps the representation simple and
    avoids pulling in a full boolean geometry library.
    """
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
    cell_is_obstacle: list[list[bool]] = []
    for y0, y1 in zip(y_coords[:-1], y_coords[1:]):
        row: list[bool] = []
        cy = (y0 + y1) / 2.0
        for x0, x1 in zip(x_coords[:-1], x_coords[1:]):
            cx = (x0 + x1) / 2.0
            is_free = any((rect.contains_point(cx, cy) for rect in free_rectangles))
            row.append(not is_free)
        cell_is_obstacle.append(row)
    merged_cells = [row[:] for row in cell_is_obstacle]
    obstacles: list[RectangleObstacle] = []
    obstacle_index = 0
    for row_idx in range(len(merged_cells)):
        for col_idx in range(len(merged_cells[row_idx])):
            if not merged_cells[row_idx][col_idx]:
                continue
            end_col = col_idx + 1
            while end_col < len(merged_cells[row_idx]) and merged_cells[row_idx][end_col]:
                end_col += 1
            end_row = row_idx + 1
            while end_row < len(merged_cells):
                if not all((merged_cells[end_row][col] for col in range(col_idx, end_col))):
                    break
                end_row += 1
            for clear_row in range(row_idx, end_row):
                for clear_col in range(col_idx, end_col):
                    merged_cells[clear_row][clear_col] = False
            obstacles.append(
                RectangleObstacle(
                    name=f"obs_{obstacle_index:02d}",
                    xmin=x_coords[col_idx],
                    xmax=x_coords[end_col],
                    ymin=y_coords[row_idx],
                    ymax=y_coords[end_row],
                )
            )
            obstacle_index += 1
    return tuple(obstacles)


def inset_rectangle_region(
    region: RectangleRegion, *, name: str, x_margin: float, y_margin: float
) -> RectangleRegion:
    """Create a smaller inner rectangle used for event-style region checks."""
    if x_margin < 0.0 or y_margin < 0.0:
        raise ValueError("x_margin and y_margin must be non-negative.")
    if x_margin * 2.0 >= region.width or y_margin * 2.0 >= region.height:
        raise ValueError(
            f"Inset margins are too large for region {region.name}: width={region.width}, height={region.height}, x_margin={x_margin}, y_margin={y_margin}."
        )
    return RectangleRegion(
        name=name,
        xmin=region.xmin + x_margin,
        xmax=region.xmax - x_margin,
        ymin=region.ymin + y_margin,
        ymax=region.ymax - y_margin,
    )


def build_default_map_config() -> MapConfig:
    """Centralized geometry for the four-start implicit-cue braided map.

    Topology:
    - Four task-specific start rooms on the left encode the task implicitly.
    - The starts merge into one long shared snake corridor.
    - Two serial binary decisions (H1/H2) are separated by long shared segments.
    - The final corridor fans out to four goal rooms on the right.
    """
    workspace = RectangleRegion("workspace", xmin=0.0, xmax=100.0, ymin=0.0, ymax=60.0)
    start_s00 = RectangleRegion("S00", xmin=2.0, xmax=8.0, ymin=44.0, ymax=50.0)
    start_s01 = RectangleRegion("S01", xmin=2.0, xmax=8.0, ymin=34.0, ymax=40.0)
    start_s10 = RectangleRegion("S10", xmin=2.0, xmax=8.0, ymin=20.0, ymax=26.0)
    start_s11 = RectangleRegion("S11", xmin=2.0, xmax=8.0, ymin=10.0, ymax=16.0)
    task_start_regions = (start_s00, start_s01, start_s10, start_s11)
    start_00_prefix = RectangleRegion("start_00_prefix", xmin=8.0, xmax=12.0, ymin=45.0, ymax=49.0)
    start_01_prefix = RectangleRegion("start_01_prefix", xmin=8.0, xmax=12.0, ymin=35.0, ymax=39.0)
    start_10_prefix = RectangleRegion("start_10_prefix", xmin=8.0, xmax=12.0, ymin=21.0, ymax=25.0)
    start_11_prefix = RectangleRegion("start_11_prefix", xmin=8.0, xmax=12.0, ymin=11.0, ymax=15.0)
    start_collector = RectangleRegion("start_collector", xmin=11.0, xmax=15.0, ymin=11.0, ymax=49.0)
    shared_entry = RectangleRegion("shared_entry", xmin=14.0, xmax=18.0, ymin=27.0, ymax=33.0)
    shared_snake_1 = RectangleRegion("shared_snake_1", xmin=18.0, xmax=24.0, ymin=27.0, ymax=33.0)
    shared_snake_2 = RectangleRegion("shared_snake_2", xmin=21.0, xmax=25.0, ymin=27.0, ymax=43.0)
    shared_snake_3 = RectangleRegion("shared_snake_3", xmin=24.0, xmax=30.0, ymin=37.0, ymax=43.0)
    shared_snake_4 = RectangleRegion("shared_snake_4", xmin=29.0, xmax=33.0, ymin=23.0, ymax=43.0)
    shared_snake_5 = RectangleRegion("shared_snake_5", xmin=31.0, xmax=35.0, ymin=23.0, ymax=29.0)
    h1_hub = RectangleRegion("H1_hub", xmin=33.0, xmax=39.0, ymin=19.0, ymax=37.0)
    branch1_upper_top = RectangleRegion(
        "branch1_upper_top", xmin=40.0, xmax=48.0, ymin=41.0, ymax=47.0
    )
    branch1_upper_rise = RectangleRegion(
        "branch1_upper_rise", xmin=37.0, xmax=41.0, ymin=33.0, ymax=47.0
    )
    branch1_upper_drop = RectangleRegion(
        "branch1_upper_drop", xmin=47.0, xmax=51.0, ymin=31.0, ymax=47.0
    )
    branch1_lower_bottom = RectangleRegion(
        "branch1_lower_bottom", xmin=40.0, xmax=48.0, ymin=9.0, ymax=15.0
    )
    branch1_lower_drop = RectangleRegion(
        "branch1_lower_drop", xmin=37.0, xmax=41.0, ymin=9.0, ymax=23.0
    )
    branch1_lower_rise = RectangleRegion(
        "branch1_lower_rise", xmin=47.0, xmax=51.0, ymin=9.0, ymax=29.0
    )
    merge1_hub = RectangleRegion("merge1_hub", xmin=49.0, xmax=55.0, ymin=23.0, ymax=37.0)
    middle_snake_1 = RectangleRegion("middle_snake_1", xmin=54.0, xmax=60.0, ymin=27.0, ymax=33.0)
    middle_snake_2 = RectangleRegion("middle_snake_2", xmin=59.0, xmax=63.0, ymin=27.0, ymax=41.0)
    middle_snake_3 = RectangleRegion("middle_snake_3", xmin=62.0, xmax=68.0, ymin=35.0, ymax=41.0)
    middle_snake_4 = RectangleRegion("middle_snake_4", xmin=67.0, xmax=71.0, ymin=17.0, ymax=41.0)
    middle_snake_5 = RectangleRegion("middle_snake_5", xmin=69.0, xmax=73.0, ymin=17.0, ymax=23.0)
    h2_hub = RectangleRegion("H2_hub", xmin=71.0, xmax=77.0, ymin=13.0, ymax=37.0)
    branch2_upper_top = RectangleRegion(
        "branch2_upper_top", xmin=78.0, xmax=86.0, ymin=39.0, ymax=45.0
    )
    branch2_upper_rise = RectangleRegion(
        "branch2_upper_rise", xmin=75.0, xmax=79.0, ymin=31.0, ymax=45.0
    )
    branch2_upper_drop = RectangleRegion(
        "branch2_upper_drop", xmin=85.0, xmax=89.0, ymin=30.0, ymax=45.0
    )
    branch2_lower_bottom = RectangleRegion(
        "branch2_lower_bottom", xmin=78.0, xmax=86.0, ymin=7.0, ymax=13.0
    )
    branch2_lower_drop = RectangleRegion(
        "branch2_lower_drop", xmin=75.0, xmax=79.0, ymin=7.0, ymax=17.0
    )
    branch2_lower_rise = RectangleRegion(
        "branch2_lower_rise", xmin=85.0, xmax=89.0, ymin=7.0, ymax=29.0
    )
    merge2_hub = RectangleRegion("merge2_hub", xmin=87.0, xmax=93.0, ymin=23.0, ymax=37.0)
    final_corridor = RectangleRegion("final_corridor", xmin=91.0, xmax=97.0, ymin=27.0, ymax=33.0)
    terminal_hub = RectangleRegion("terminal_hub", xmin=95.0, xmax=99.0, ymin=9.0, ymax=51.0)
    goal_regions = (
        GoalRegion("G00", xmin=98.0, xmax=100.0, ymin=42.0, ymax=50.0),
        GoalRegion("G01", xmin=98.0, xmax=100.0, ymin=32.0, ymax=38.0),
        GoalRegion("G10", xmin=98.0, xmax=100.0, ymin=22.0, ymax=28.0),
        GoalRegion("G11", xmin=98.0, xmax=100.0, ymin=10.0, ymax=16.0),
    )
    decision_points = (DecisionPoint("H1", x=36.0, y=28.0), DecisionPoint("H2", x=74.0, y=25.0))
    start_region = SemanticRegion(
        "start_region",
        rectangles=(
            *task_start_regions,
            start_00_prefix,
            start_01_prefix,
            start_10_prefix,
            start_11_prefix,
            start_collector,
        ),
    )
    shared_corridor_region = SemanticRegion(
        "shared_corridor_region",
        rectangles=(
            shared_entry,
            shared_snake_1,
            shared_snake_2,
            shared_snake_3,
            shared_snake_4,
            shared_snake_5,
        ),
    )
    decision_region_h1 = SemanticRegion("decision_region_H1", rectangles=(h1_hub,))
    branch1_upper_region = SemanticRegion(
        "branch1_upper_region",
        rectangles=(branch1_upper_top, branch1_upper_rise, branch1_upper_drop),
    )
    branch1_lower_region = SemanticRegion(
        "branch1_lower_region",
        rectangles=(branch1_lower_bottom, branch1_lower_drop, branch1_lower_rise),
    )
    branch1_upper_detection_region = inset_rectangle_region(
        branch1_upper_top, name="branch1_upper_detection_region", x_margin=2.0, y_margin=1.0
    )
    branch1_lower_detection_region = inset_rectangle_region(
        branch1_lower_bottom, name="branch1_lower_detection_region", x_margin=2.0, y_margin=1.0
    )
    merge_region_1 = SemanticRegion("merge_region_1", rectangles=(merge1_hub,))
    middle_corridor_region = SemanticRegion(
        "middle_corridor_region",
        rectangles=(middle_snake_1, middle_snake_2, middle_snake_3, middle_snake_4, middle_snake_5),
    )
    decision_region_h2 = SemanticRegion("decision_region_H2", rectangles=(h2_hub,))
    branch2_upper_region = SemanticRegion(
        "branch2_upper_region",
        rectangles=(branch2_upper_top, branch2_upper_rise, branch2_upper_drop),
    )
    branch2_lower_region = SemanticRegion(
        "branch2_lower_region",
        rectangles=(branch2_lower_bottom, branch2_lower_drop, branch2_lower_rise),
    )
    branch2_upper_detection_region = inset_rectangle_region(
        branch2_upper_top, name="branch2_upper_detection_region", x_margin=2.0, y_margin=1.0
    )
    branch2_lower_detection_region = inset_rectangle_region(
        branch2_lower_bottom, name="branch2_lower_detection_region", x_margin=2.0, y_margin=1.0
    )
    merge_region_2 = SemanticRegion("merge_region_2", rectangles=(merge2_hub,))
    final_corridor_region = SemanticRegion(
        "final_corridor_region", rectangles=(final_corridor, terminal_hub)
    )
    free_space_rectangles = (
        *task_start_regions,
        start_00_prefix,
        start_01_prefix,
        start_10_prefix,
        start_11_prefix,
        start_collector,
        shared_entry,
        shared_snake_1,
        shared_snake_2,
        shared_snake_3,
        shared_snake_4,
        shared_snake_5,
        h1_hub,
        branch1_upper_rise,
        branch1_upper_top,
        branch1_upper_drop,
        branch1_lower_drop,
        branch1_lower_bottom,
        branch1_lower_rise,
        merge1_hub,
        middle_snake_1,
        middle_snake_2,
        middle_snake_3,
        middle_snake_4,
        middle_snake_5,
        h2_hub,
        branch2_upper_rise,
        branch2_upper_top,
        branch2_upper_drop,
        branch2_lower_drop,
        branch2_lower_bottom,
        branch2_lower_rise,
        merge2_hub,
        final_corridor,
        terminal_hub,
        *goal_regions,
    )
    obstacle_rectangles = build_obstacles_from_free_rectangles(workspace, free_space_rectangles)
    path_labels = (
        ("start cue", (8.0, 53.0)),
        ("shared trunk 1", (25.0, 46.0)),
        ("decision 1", (39.0, 50.0)),
        ("shared trunk 2", (63.0, 44.0)),
        ("decision 2", (82.0, 48.0)),
        ("goal fan-out", (97.0, 54.0)),
    )
    return MapConfig(
        workspace=workspace,
        start_region=start_region,
        task_start_regions=task_start_regions,
        shared_corridor_region=shared_corridor_region,
        decision_region_h1=decision_region_h1,
        branch1_upper_region=branch1_upper_region,
        branch1_lower_region=branch1_lower_region,
        branch1_upper_detection_region=branch1_upper_detection_region,
        branch1_lower_detection_region=branch1_lower_detection_region,
        merge_region_1=merge_region_1,
        middle_corridor_region=middle_corridor_region,
        decision_region_h2=decision_region_h2,
        branch2_upper_region=branch2_upper_region,
        branch2_lower_region=branch2_lower_region,
        branch2_upper_detection_region=branch2_upper_detection_region,
        branch2_lower_detection_region=branch2_lower_detection_region,
        merge_region_2=merge_region_2,
        final_corridor_region=final_corridor_region,
        free_space_rectangles=free_space_rectangles,
        obstacle_rectangles=obstacle_rectangles,
        goal_regions=goal_regions,
        terminal_regions=goal_regions,
        decision_points=decision_points,
        path_labels=path_labels,
    )


@lru_cache(maxsize=1)
def get_default_map_config() -> MapConfig:
    """Return a cached default map config for helper functions."""
    return build_default_map_config()


def _resolve_config(config: MapConfig | None) -> MapConfig:
    return get_default_map_config() if config is None else config


def build_task_spec(task_id: int) -> TaskSpec:
    """Map a task id in {0,1,2,3} to a 2-bit code and goal name."""
    if task_id not in TASK_ID_TO_GOAL_NAME:
        raise ValueError(
            f"Unsupported task_id={task_id}. Expected one of {sorted(TASK_ID_TO_GOAL_NAME)}."
        )
    task_code = format(task_id, "02b")
    return TaskSpec(
        task_id=task_id,
        task_code=task_code,
        task_bits=(int(task_code[0]), int(task_code[1])),
        start_region_name=f"S{task_code}",
        target_goal_name=TASK_ID_TO_GOAL_NAME[task_id],
    )


def get_task_start_region(task_id: int, config: MapConfig | None = None) -> RectangleRegion:
    """Return the task-specific implicit-cue start room for a task id."""
    resolved_config = _resolve_config(config)
    start_region_name = build_task_spec(task_id).start_region_name
    for start_region in resolved_config.task_start_regions:
        if start_region.name == start_region_name:
            return start_region
    raise RuntimeError(f"Start region {start_region_name} for task_id={task_id} was not found.")


def is_in_bounds(x: float, y: float, config: MapConfig | None = None) -> bool:
    """Check whether a point lies inside the global workspace.

    Boundary convention:
    - The workspace is treated as a closed set, so boundary points are in-bounds.
    """
    resolved_config = _resolve_config(config)
    return resolved_config.workspace.contains_point(x, y)


def is_in_obstacle(x: float, y: float, config: MapConfig | None = None) -> bool:
    """Check whether a point lies inside any obstacle rectangle.

    Boundary convention:
    - Obstacles are treated as closed sets, so points on obstacle edges count
      as collisions. This makes corridor walls conservative and unambiguous.
    """
    resolved_config = _resolve_config(config)
    return any((obstacle.contains_point(x, y) for obstacle in resolved_config.obstacle_rectangles))


def is_state_valid(x: float, y: float, config: MapConfig | None = None) -> bool:
    """Check whether a point-robot state is valid in the current map."""
    return is_in_bounds(x, y, config=config) and (not is_in_obstacle(x, y, config=config))


def sample_workspace_state(
    rng: SupportsUniform, config: MapConfig | None = None
) -> tuple[float, float]:
    """Sample a point uniformly from the workspace bounding box."""
    resolved_config = _resolve_config(config)
    workspace = resolved_config.workspace
    return (
        float(rng.uniform(workspace.xmin, workspace.xmax)),
        float(rng.uniform(workspace.ymin, workspace.ymax)),
    )


def sample_free_state(
    rng: SupportsUniform,
    config: MapConfig | None = None,
    max_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS,
) -> tuple[float, float]:
    """Rejection-sample one valid point-robot state from free space."""
    resolved_config = _resolve_config(config)
    for _ in range(max_attempts):
        x, y = sample_workspace_state(rng, config=resolved_config)
        if is_state_valid(x, y, config=resolved_config):
            return (x, y)
    raise RuntimeError(
        "Failed to sample a valid free state within the attempt budget. Check the map geometry or increase max_attempts."
    )


def sample_valid_state_in_region(
    region: RectangleRegion,
    rng: SupportsUniform,
    config: MapConfig | None = None,
    max_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS,
) -> tuple[float, float]:
    """Sample a valid state constrained to a specific region."""
    resolved_config = _resolve_config(config)
    for _ in range(max_attempts):
        x = float(rng.uniform(region.xmin, region.xmax))
        y = float(rng.uniform(region.ymin, region.ymax))
        if is_state_valid(x, y, config=resolved_config):
            return (x, y)
    raise RuntimeError(
        f"Failed to sample a valid state from region {region.name}. Check whether the region overlaps free space."
    )


def get_goal_region_for_state(
    x: float, y: float, config: MapConfig | None = None
) -> GoalRegion | None:
    """Return the goal region containing the current point, if any."""
    resolved_config = _resolve_config(config)
    for goal_region in resolved_config.goal_regions:
        if goal_region.contains_point(x, y):
            return goal_region
    return None


def get_phase_name(x: float, y: float, config: MapConfig | None = None) -> str:
    """Return the semantic phase label for a 2D point.

    Priority is chosen so branch and merge events remain stable near region
    boundaries: terminal > final > merge2 > branch2 > H2 > middle > merge1 >
    branch1 > H1 > start > shared corridor.
    """
    resolved_config = _resolve_config(config)
    if not is_in_bounds(x, y, config=resolved_config):
        return "out_of_bounds"
    if is_in_obstacle(x, y, config=resolved_config):
        return "obstacle"
    terminal_region = get_goal_region_for_state(x, y, config=resolved_config)
    if terminal_region is not None:
        return terminal_region.name
    if resolved_config.final_corridor_region.contains_point(x, y):
        return resolved_config.final_corridor_region.name
    if resolved_config.merge_region_2.contains_point(x, y):
        return resolved_config.merge_region_2.name
    if resolved_config.branch2_upper_region.contains_point(x, y):
        return resolved_config.branch2_upper_region.name
    if resolved_config.branch2_lower_region.contains_point(x, y):
        return resolved_config.branch2_lower_region.name
    if resolved_config.decision_region_h2.contains_point(x, y):
        return resolved_config.decision_region_h2.name
    if resolved_config.middle_corridor_region.contains_point(x, y):
        return resolved_config.middle_corridor_region.name
    if resolved_config.merge_region_1.contains_point(x, y):
        return resolved_config.merge_region_1.name
    if resolved_config.branch1_upper_region.contains_point(x, y):
        return resolved_config.branch1_upper_region.name
    if resolved_config.branch1_lower_region.contains_point(x, y):
        return resolved_config.branch1_lower_region.name
    if resolved_config.decision_region_h1.contains_point(x, y):
        return resolved_config.decision_region_h1.name
    if resolved_config.start_region.contains_point(x, y):
        return "start_region"
    if resolved_config.shared_corridor_region.contains_point(x, y):
        return resolved_config.shared_corridor_region.name
    return "free_space_other"


def get_branch_detection_event(
    x: float, y: float, config: MapConfig | None = None
) -> tuple[str, str] | None:
    """Return a stage/phase tuple only inside the narrowed branch detection windows."""
    resolved_config = _resolve_config(config)
    detection_regions = (
        ("H1", "branch1_upper_region", resolved_config.branch1_upper_detection_region),
        ("H1", "branch1_lower_region", resolved_config.branch1_lower_detection_region),
        ("H2", "branch2_upper_region", resolved_config.branch2_upper_detection_region),
        ("H2", "branch2_lower_region", resolved_config.branch2_lower_detection_region),
    )
    for stage_name, phase_name, detection_region in detection_regions:
        if detection_region.contains_point(x, y):
            return (stage_name, phase_name)
    return None


def _first_index_of_phase(phase_labels: tuple[str, ...], target_phase: str) -> int | None:
    for index, phase_label in enumerate(phase_labels):
        if phase_label == target_phase:
            return index
    return None


def annotate_trajectory_phases(
    path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig | None = None,
) -> TrajectoryPhaseAnnotation:
    """Annotate each trajectory point with a semantic phase label."""
    phase_labels = tuple((get_phase_name(x, y, config=config) for x, y in path))
    first_h1_index = _first_index_of_phase(phase_labels, "decision_region_H1")
    first_h2_index = _first_index_of_phase(phase_labels, "decision_region_H2")
    first_terminal_index = None
    first_terminal_phase = None
    resolved_config = _resolve_config(config)
    terminal_phase_names = {goal_region.name for goal_region in resolved_config.terminal_regions}
    for index, phase_label in enumerate(phase_labels):
        if phase_label in terminal_phase_names:
            first_terminal_index = index
            first_terminal_phase = phase_label
            break
    return TrajectoryPhaseAnnotation(
        phase_labels=phase_labels,
        first_h1_index=first_h1_index,
        first_h2_index=first_h2_index,
        first_terminal_index=first_terminal_index,
        first_terminal_phase=first_terminal_phase,
    )


def plot_map(config: MapConfig, show: bool = True, ax=None):
    """Plot the rectangle-based 2D map."""
    if plt is None or Line2D is None or Patch is None:
        raise RuntimeError(
            "matplotlib is required to plot this map. Install it first, for example: pip install matplotlib"
        ) from _MATPLOTLIB_IMPORT_ERROR
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 7))
    else:
        fig = ax.figure
        ax.clear()
    workspace_patch = config.workspace.as_patch(
        facecolor="#f8f4e3", edgecolor="black", linewidth=2.5, zorder=0
    )
    ax.add_patch(workspace_patch)
    for obstacle in config.obstacle_rectangles:
        ax.add_patch(
            obstacle.as_patch(facecolor="#4a4a4a", edgecolor="#2f2f2f", linewidth=0.8, zorder=1)
        )
    for start_region in config.task_start_regions:
        ax.add_patch(
            start_region.as_patch(
                facecolor="#4c78a8", edgecolor="#1f3552", linewidth=1.2, alpha=0.95, zorder=3
            )
        )
        start_x, start_y = start_region.center
        ax.text(
            start_x,
            start_y,
            start_region.name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=4,
        )
    goal_colors = {"G00": "#1b9e77", "G01": "#66a61e", "G10": "#d95f02", "G11": "#e6ab02"}
    for goal in config.goal_regions:
        ax.add_patch(
            goal.as_patch(
                facecolor=goal_colors[goal.name],
                edgecolor="black",
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
            )
        )
        goal_x, goal_y = goal.center
        ax.text(
            goal_x,
            goal_y,
            goal.name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=4,
        )
    for label, (x, y) in config.path_labels:
        ax.text(x, y, label, fontsize=10, color="#5b5b5b", ha="center", va="center", zorder=2)
    legend_handles = build_base_legend_handles()
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)
    ax.set_xlim(config.workspace.xmin - 1.0, config.workspace.xmax + 1.0)
    ax.set_ylim(config.workspace.ymin - 1.0, config.workspace.ymax + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "Four-Start Implicit-Cue Map with Shared Snake Corridor and Two Serial Binary Branches"
    )
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.4)
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax)


def build_base_legend_handles():
    """Legend entries shared by the static map plot and debug plots."""
    if Line2D is None or Patch is None:
        raise RuntimeError(
            "matplotlib is required to build legend handles. Install it first, for example: pip install matplotlib"
        ) from _MATPLOTLIB_IMPORT_ERROR
    return [
        Patch(facecolor="#4a4a4a", edgecolor="#2f2f2f", label="Obstacle"),
        Patch(facecolor="#4c78a8", edgecolor="#1f3552", label="Task Start Room"),
        Patch(facecolor="#1b9e77", edgecolor="black", label="Goal Region"),
    ]


def plot_trajectory_with_phases(
    path: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: MapConfig | None = None,
    show: bool = True,
):
    """Plot a trajectory and highlight the semantic phases it traverses."""
    if plt is None or Line2D is None:
        raise RuntimeError(
            "matplotlib is required to plot trajectory phases. Install it first, for example: pip install matplotlib"
        ) from _MATPLOTLIB_IMPORT_ERROR
    resolved_config = _resolve_config(config)
    annotation = annotate_trajectory_phases(path, config=resolved_config)
    fig, ax = plot_map(resolved_config, show=False)
    if path:
        path_xs = [point[0] for point in path]
        path_ys = [point[1] for point in path]
        ax.plot(path_xs, path_ys, color="#333333", linewidth=1.8, alpha=0.65, zorder=5)
    phase_points: dict[str, list[tuple[float, float]]] = {}
    for point, phase_label in zip(path, annotation.phase_labels, strict=False):
        phase_points.setdefault(phase_label, []).append(point)
    for phase_label, points in phase_points.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.scatter(
            xs,
            ys,
            s=28,
            c=PHASE_COLOR_BY_NAME.get(phase_label, "#666666"),
            alpha=0.9,
            linewidths=0.0,
            zorder=6,
        )
    event_specs = [
        ("H1", annotation.first_h1_index, "#c0392b"),
        ("H2", annotation.first_h2_index, "#ff7f0e"),
        ("T", annotation.first_terminal_index, "#111111"),
    ]
    for event_name, event_index, color in event_specs:
        if event_index is None:
            continue
        event_x, event_y = path[event_index]
        ax.scatter(
            event_x,
            event_y,
            s=120,
            c=color,
            marker="X",
            edgecolors="white",
            linewidths=1.0,
            zorder=8,
        )
        ax.text(
            event_x + 1.2,
            event_y + 1.2,
            f"{event_name}@{event_index}",
            fontsize=9,
            color=color,
            fontweight="bold",
            zorder=9,
        )
    phase_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PHASE_COLOR_BY_NAME.get(phase_label, "#666666"),
            markeredgecolor=PHASE_COLOR_BY_NAME.get(phase_label, "#666666"),
            markersize=7,
            label=phase_label,
        )
        for phase_label in phase_points
    ]
    event_handles = [
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#c0392b",
            markeredgecolor="white",
            markersize=8,
            label="First H1 Entry",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#ff7f0e",
            markeredgecolor="white",
            markersize=8,
            label="First H2 Entry",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#111111",
            markeredgecolor="white",
            markersize=8,
            label="First Terminal Entry",
        ),
    ]
    ax.legend(
        handles=build_base_legend_handles() + phase_handles + event_handles,
        loc="upper left",
        frameon=True,
    )
    ax.set_title(
        f"Trajectory Phase Annotation\nfirst_h1={annotation.first_h1_index}, first_h2={annotation.first_h2_index}, first_terminal={annotation.first_terminal_index} ({annotation.first_terminal_phase})"
    )
    fig.tight_layout()
    if show:
        plt.show()
    return (fig, ax, annotation)


class BraidedHub2DEnv:
    """Lightweight 2D point-robot environment for planning and data generation."""

    def __init__(
        self,
        map_config: MapConfig | None = None,
        rng_seed: int = DEFAULT_RANDOM_SEED,
        dt: float = DEFAULT_DT,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        enable_randomize: bool = False,
    ) -> None:
        self.map_config = _resolve_config(map_config)
        self.start_region = self.map_config.start_region
        self.goal_regions = self.map_config.goal_regions
        self.goal_region_by_name = {
            goal_region.name: goal_region for goal_region in self.goal_regions
        }
        self.dt = float(dt)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.enable_randomize = bool(enable_randomize)
        self.rng = random.Random(rng_seed)
        self.state: tuple[float, float] | None = None
        self.trajectory: list[tuple[float, float]] = []
        self.step_count = 0
        self.episode_task_id: int | None = None
        self.task_spec: TaskSpec | None = None
        self.start_region_name: str | None = None
        self.target_goal_name: str | None = None
        self.target_goal_region: GoalRegion | None = None
        self.last_info: dict[str, Any] = {}
        self.done = False
        self._render_fig = None
        self._render_ax = None

    def _resolve_start_state(
        self, task_start_region: RectangleRegion, enable_randomize: bool
    ) -> tuple[float, float]:
        """Resolve the start state for one reset.

        By default the start is the region center for reproducibility.
        Randomized start sampling remains available behind an explicit flag.
        """
        if enable_randomize:
            return sample_valid_state_in_region(
                task_start_region, rng=self.rng, config=self.map_config
            )
        center_state = (float(task_start_region.center[0]), float(task_start_region.center[1]))
        if not is_state_valid(*center_state, config=self.map_config):
            raise RuntimeError(
                f"The deterministic start-region center is not a valid free-space state. region={task_start_region.name}, center={center_state}. Fix the map geometry or call reset(enable_randomize=True)."
            )
        return center_state

    def reset(
        self, task_id: int | None = None, enable_randomize: bool | None = None
    ) -> tuple[float, float]:
        """Start a new episode from the task start-region center by default."""
        chosen_task_id = self.sample_task_id() if task_id is None else int(task_id)
        self.task_spec = build_task_spec(chosen_task_id)
        self.episode_task_id = self.task_spec.task_id
        self.start_region_name = self.task_spec.start_region_name
        self.target_goal_name = self.task_spec.target_goal_name
        self.target_goal_region = self.goal_region_by_name[self.target_goal_name]
        task_start_region = get_task_start_region(chosen_task_id, self.map_config)
        use_randomized_start = (
            self.enable_randomize if enable_randomize is None else bool(enable_randomize)
        )
        self.state = self._resolve_start_state(
            task_start_region=task_start_region, enable_randomize=use_randomized_start
        )
        self.trajectory = [self.state]
        self.step_count = 0
        self.done = False
        self.last_info = {
            "task_id": self.episode_task_id,
            "task_code": self.task_spec.task_code,
            "task_bits": self.task_spec.task_bits,
            "start_region_name": self.start_region_name,
            "target_goal_name": self.target_goal_name,
            "reached_goal": None,
            "success": False,
            "collision_mode": "reject",
            "collision_detected": False,
            "collision_rejected": False,
            "start_randomized": use_randomized_start,
        }
        return self.state

    def sample_task_id(self) -> int:
        """Uniformly sample one of the four task ids."""
        return int(self.rng.choice(tuple(TASK_ID_TO_GOAL_NAME)))

    def is_cue_visible(self) -> bool:
        """The explicit cue channel is disabled in the implicit-cue map."""
        return False

    def get_cue_payload(self) -> dict[str, Any] | None:
        """Explicit cue payload is disabled; task information is carried by path history."""
        return None

    def get_full_observation(self) -> dict[str, Any]:
        """Return full observation including internal task truth."""
        if self.state is None or self.task_spec is None or self.target_goal_region is None:
            raise RuntimeError("Call reset() before requesting observations.")
        return {
            "x": self.state[0],
            "y": self.state[1],
            "position": self.state,
            "step_count": self.step_count,
            "task_id": self.task_spec.task_id,
            "task_code": self.task_spec.task_code,
            "task_bits": self.task_spec.task_bits,
            "start_region_name": self.start_region_name,
            "target_goal_name": self.task_spec.target_goal_name,
            "target_goal_center": self.target_goal_region.center,
        }

    def get_partial_observation(self) -> dict[str, Any]:
        """Return the policy-visible observation without any explicit cue channel."""
        if self.state is None:
            raise RuntimeError("Call reset() before requesting observations.")
        return {
            "x": self.state[0],
            "y": self.state[1],
            "position": self.state,
            "step_count": self.step_count,
        }

    def is_state_valid(self, state: tuple[float, float]) -> bool:
        """Planner-friendly state validity wrapper bound to this map instance."""
        return is_state_valid(*state, config=self.map_config)

    def sample_free_state(self) -> tuple[float, float]:
        """Planner-friendly free-space sampler bound to this map instance."""
        return sample_free_state(self.rng, config=self.map_config)

    def get_goal_region(self, state: tuple[float, float]) -> GoalRegion | None:
        """Return any terminal goal region containing the queried state."""
        return get_goal_region_for_state(*state, config=self.map_config)

    def get_phase_name(self, state: tuple[float, float]) -> str:
        """Return the semantic phase label for a queried state."""
        return get_phase_name(*state, config=self.map_config)

    def annotate_trajectory_phases(
        self, path: list[tuple[float, float]] | tuple[tuple[float, float], ...]
    ) -> TrajectoryPhaseAnnotation:
        """Annotate a path using this environment's semantic map."""
        return annotate_trajectory_phases(path, config=self.map_config)

    def plot_trajectory_with_phases(
        self, path: list[tuple[float, float]] | tuple[tuple[float, float], ...], show: bool = True
    ):
        """Convenience wrapper around the module-level phase plotter."""
        return plot_trajectory_with_phases(path, config=self.map_config, show=show)

    def step(
        self, action: tuple[float, float]
    ) -> tuple[tuple[float, float], float, bool, dict[str, Any]]:
        """Advance the robot, rejecting invalid transitions and rewarding the task-selected goal."""
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new one.")
        if (
            self.task_spec is None
            or self.target_goal_region is None
            or self.target_goal_name is None
        ):
            raise RuntimeError("Call reset() before step().")
        dx, dy = action
        proposed_state = (float(self.state[0] + dx * self.dt), float(self.state[1] + dy * self.dt))
        proposed_valid = is_state_valid(*proposed_state, config=self.map_config)
        collision_detected = not proposed_valid
        if proposed_valid:
            collision_rejected = False
            next_state = proposed_state
        else:
            collision_rejected = True
            next_state = self.state
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
            "task_id": self.episode_task_id,
            "task_code": self.task_spec.task_code,
            "task_bits": self.task_spec.task_bits,
            "start_region_name": self.start_region_name,
            "target_goal_name": self.target_goal_name,
            "step_count": self.step_count,
            "action": (float(dx), float(dy)),
            "collision_mode": "reject",
            "proposed_state": proposed_state,
            "applied_state": next_state,
            "proposed_state_valid": proposed_valid,
            "collision_detected": collision_detected,
            "collision_rejected": collision_rejected,
            "reached_goal": None if reached_goal is None else reached_goal.name,
            "success": success,
            "partial_observation": self.get_partial_observation(),
            "full_observation": self.get_full_observation(),
        }
        self.last_info = info
        return (next_state, reward, self.done, info)

    def render(
        self,
        show_trajectory: bool = True,
        show: bool = True,
        refresh_every_n_steps: int = 1,
        pause_seconds: float = DEFAULT_RENDER_PAUSE_SECONDS,
        force: bool = False,
    ):
        """Render the environment and current point robot state.

        Set refresh_every_n_steps > 1 to only redraw every N environment steps.
        This gives a simple animation mode without introducing a separate viewer.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before render().")
        if plt is None:
            raise RuntimeError(
                "matplotlib is required to render this environment. Install it first, for example: pip install matplotlib"
            ) from _MATPLOTLIB_IMPORT_ERROR
        should_refresh = force or self.step_count == 0 or refresh_every_n_steps <= 1
        if not should_refresh:
            should_refresh = self.step_count % refresh_every_n_steps == 0
        if not should_refresh:
            return (self._render_fig, self._render_ax)
        if self._render_fig is None or self._render_ax is None:
            self._render_fig, self._render_ax = plt.subplots(figsize=(13, 7))
        elif not plt.fignum_exists(self._render_fig.number):
            self._render_fig, self._render_ax = plt.subplots(figsize=(13, 7))
        fig, ax = plot_map(self.map_config, show=False, ax=self._render_ax)
        if self.target_goal_region is not None:
            ax.add_patch(
                self.target_goal_region.as_patch(
                    facecolor="none", edgecolor="#111111", linewidth=2.5, linestyle="--", zorder=6
                )
            )
        if show_trajectory and len(self.trajectory) >= 2:
            traj_xs = [point[0] for point in self.trajectory]
            traj_ys = [point[1] for point in self.trajectory]
            ax.plot(
                traj_xs,
                traj_ys,
                color=VALID_SAMPLE_COLOR,
                linewidth=TRAJECTORY_LINEWIDTH,
                alpha=0.85,
                zorder=6,
            )
        ax.scatter(
            self.state[0],
            self.state[1],
            s=ROBOT_MARKER_SIZE,
            c="#111111",
            edgecolors="white",
            linewidths=1.2,
            zorder=7,
        )
        ax.set_title(
            f"BraidedHub2DEnv\nstep={self.step_count}, task_id={self.episode_task_id}, code={(None if self.task_spec is None else self.task_spec.task_code)}, start={self.start_region_name}, target={self.target_goal_name}, state=({self.state[0]:.2f}, {self.state[1]:.2f})"
        )
        fig.tight_layout()
        if show:
            plt.show(block=False)
            plt.pause(pause_seconds)
        return (fig, ax)

    def rollout_with_random_actions(
        self,
        num_steps: int = DEFAULT_RANDOM_ROLLOUT_STEPS,
        action_scale: float = DEFAULT_RANDOM_ACTION_SCALE,
        render_every_n_steps: int = DEFAULT_RENDER_EVERY_N_STEPS,
        pause_seconds: float = DEFAULT_RENDER_PAUSE_SECONDS,
        show: bool = True,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Run one random rollout for quick collision and wall-debug checks."""
        initial_state = self.reset(task_id=task_id)
        total_reward = 0.0
        collision_count = 0
        actions: list[tuple[float, float]] = []
        rewards: list[float] = []
        if show:
            self.render(
                show_trajectory=True,
                show=True,
                refresh_every_n_steps=1,
                pause_seconds=pause_seconds,
                force=True,
            )
        for _ in range(num_steps):
            action = (
                float(self.rng.uniform(-action_scale, action_scale)),
                float(self.rng.uniform(-action_scale, action_scale)),
            )
            next_state, reward, done, info = self.step(action)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            if info["collision_rejected"]:
                collision_count += 1
            if show:
                self.render(
                    show_trajectory=True,
                    show=True,
                    refresh_every_n_steps=render_every_n_steps,
                    pause_seconds=pause_seconds,
                )
            if done:
                break
        if show:
            self.render(
                show_trajectory=True,
                show=True,
                refresh_every_n_steps=1,
                pause_seconds=pause_seconds,
                force=True,
            )
        return {
            "initial_state": initial_state,
            "final_state": self.state,
            "trajectory": list(self.trajectory),
            "actions": actions,
            "rewards": rewards,
            "total_reward": total_reward,
            "num_steps": self.step_count,
            "collision_count": collision_count,
            "done": self.done,
            "task_id": self.episode_task_id,
            "task_code": None if self.task_spec is None else self.task_spec.task_code,
            "target_goal_name": self.target_goal_name,
            "reached_goal": self.last_info.get("reached_goal"),
            "success": self.last_info.get("success"),
        }

    def manual_test_task_routes(
        self,
        num_resets: int = DEFAULT_MANUAL_TEST_RESETS,
        steps_per_reset: int = DEFAULT_MANUAL_TEST_STEPS,
        action_scale: float = 1.0,
    ) -> None:
        """Print task assignments and policy-visible observations without explicit cues."""
        for reset_idx in range(num_resets):
            state = self.reset()
            print(
                f"[reset {reset_idx:02d}] task_id={self.episode_task_id}, task_code={(self.task_spec.task_code if self.task_spec else None)}, task_bits={(self.task_spec.task_bits if self.task_spec else None)}, start={self.start_region_name}, target={self.target_goal_name}, state=({state[0]:.2f}, {state[1]:.2f})"
            )
            print(f"  partial_obs@reset: {self.get_partial_observation()}")
            for _ in range(steps_per_reset):
                action = (
                    float(self.rng.uniform(-action_scale, action_scale)),
                    float(self.rng.uniform(-action_scale, action_scale)),
                )
                _, reward, done, info = self.step(action)
                print(
                    f"  step={self.step_count:02d}, reward={reward:.2f}, done={done}, obs={info['partial_observation']}"
                )
                if done:
                    break

    def manual_test_task_cues(
        self,
        num_resets: int = DEFAULT_MANUAL_TEST_RESETS,
        steps_per_reset: int = DEFAULT_MANUAL_TEST_STEPS,
        action_scale: float = 1.0,
    ) -> None:
        """Backward-compatible wrapper for the old explicit-cue test entry point."""
        self.manual_test_task_routes(
            num_resets=num_resets, steps_per_reset=steps_per_reset, action_scale=action_scale
        )

    def close(self) -> None:
        """Close any cached matplotlib figure created by render()."""
        if plt is not None and self._render_fig is not None:
            plt.close(self._render_fig)
        self._render_fig = None
        self._render_ax = None


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
    task_ids = sorted(TASK_ID_TO_GOAL_NAME)
    repeated = [task_ids[index % len(task_ids)] for index in range(num_rollouts)]
    rng = np.random.default_rng(seed)
    rng.shuffle(repeated)
    return [int(task_id) for task_id in repeated]


def detect_branch_mismatch(
    task_spec, state_xy: tuple[float, float], config: MapConfig | None = None
) -> dict[str, str] | None:
    expected_branch1 = BRANCH1_PHASE_BY_BIT[int(task_spec.task_bits[0])]
    expected_branch2 = BRANCH2_PHASE_BY_BIT[int(task_spec.task_bits[1])]
    detection_event = get_branch_detection_event(
        float(state_xy[0]), float(state_xy[1]), config=config
    )
    if detection_event is None:
        return None
    stage_name, observed_phase = detection_event
    expected_phase = expected_branch1 if stage_name == "H1" else expected_branch2
    if observed_phase != expected_phase:
        return {
            "stage": stage_name,
            "expected_phase": expected_phase,
            "observed_phase": observed_phase,
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
    image_key, image_shape = resolve_single_visual_observation_feature(cfg)
    image_hw = (int(image_shape[1]), int(image_shape[2]))
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
        "[info] braidedhub eval reset mode: "
        + ("randomized starts" if enable_randomize else "deterministic center starts")
    )
    print(f"[info] braidedhub collision mode: {'reject'} " + "(invalid moves are rejected)")
    env = BraidedHub2DEnv(
        map_config=map_config, rng_seed=eval_seed, enable_randomize=enable_randomize
    )
    base_img = make_lerobot_base_image(map_config, image_size=max(image_hw))
    if base_img.shape[0] != image_hw[0] or base_img.shape[1] != image_hw[1]:
        raise RuntimeError(
            f"Rendered image size mismatch between dataset image features and eval map image. Policy expects HxW={image_hw}, base_img={base_img.shape[:2]}."
        )
    results = []
    success_count = 0
    branch_failure_count = 0
    task_success_counts = {int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)}
    task_rollout_counts = {int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)}
    task_branch_failure_counts = {int(task_id): 0 for task_id in sorted(TASK_ID_TO_GOAL_NAME)}
    task_schedule = build_balanced_task_schedule(eval_num_rollouts, eval_seed)
    for ep_idx, task_id in enumerate(task_schedule):
        task_spec = build_task_spec(task_id)
        task_rollout_counts[task_id] += 1
        state_xy = tuple((float(v) for v in env.reset(task_id=task_id)))
        if env.start_region_name != task_spec.start_region_name:
            raise RuntimeError(
                f"Environment reset returned a start region that does not match the task. task={task_spec.task_code}, expected_start={task_spec.start_region_name}, got={env.start_region_name}"
            )
        if hasattr(policy, "reset"):
            policy.reset()
        episode_reward = 0.0
        episode_collision_detection_count = 0
        episode_collision_rejection_count = 0
        video_path = output_dir / f"rollout_{ep_idx:03d}_task_{task_spec.task_code}.mp4"
        writer = start_ffmpeg_raw_writer(video_path, image_hw[1], image_hw[0], eval_fps)
        if writer.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin for rollout video writing.")
        trajectory = [state_xy]
        state_history = deque()
        previous_signature_vec = None
        last_info = {
            **env.last_info,
            "phase_name": env.get_phase_name(state_xy),
            "branch_mismatch": False,
            "branch_mismatch_stage": None,
            "expected_branch_phase": None,
            "observed_branch_phase": None,
            "failure_reason": None,
        }
        for _step_idx in range(eval_max_steps):
            frame = render_lerobot_frame(base_img, config=map_config, robot_xy=state_xy)
            writer.stdin.write(frame.astype(np.uint8).tobytes())
            obs = build_eval_observation(
                state_xy=state_xy,
                rgb_frame=frame,
                state_key=state_key,
                image_key=image_key,
                state_dim=state_dim,
            )
            assert state_history is not None
            state_now = obs[state_key].detach().cpu().numpy().astype(np.float32, copy=False)
            state_history.append(state_now.copy())
            signature_vec = _compute_online_signature_prefix(
                state_history=state_history,
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
            action_np = action.squeeze(0).detach().cpu().numpy()
            dx = float(action_np[0]) if action_np.shape[0] >= 1 else 0.0
            dy = float(action_np[1]) if action_np.shape[0] >= 2 else 0.0
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > args.max_action_step and norm > 1e-08:
                scale = args.max_action_step / norm
                dx *= scale
                dy *= scale
            next_state, reward, done, info = env.step((dx, dy))
            state_xy = (float(next_state[0]), float(next_state[1]))
            trajectory.append(state_xy)
            episode_reward += float(reward)
            if info.get("collision_detected", False):
                episode_collision_detection_count += 1
            if info.get("collision_rejected", False):
                episode_collision_rejection_count += 1
            phase_name = env.get_phase_name(state_xy)
            mismatch = detect_branch_mismatch(
                task_spec=task_spec, state_xy=state_xy, config=map_config
            )
            last_info = {
                **info,
                "phase_name": phase_name,
                "branch_mismatch": False,
                "branch_mismatch_stage": None,
                "expected_branch_phase": None,
                "observed_branch_phase": None,
                "failure_reason": None,
            }
            if mismatch is not None:
                last_info = {
                    **last_info,
                    "branch_mismatch": True,
                    "branch_mismatch_stage": mismatch["stage"],
                    "expected_branch_phase": mismatch["expected_phase"],
                    "observed_branch_phase": mismatch["observed_phase"],
                    "failure_reason": "wrong_branch",
                    "success": False,
                }
                final_frame = render_lerobot_frame(base_img, config=map_config, robot_xy=state_xy)
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break
            if done:
                final_frame = render_lerobot_frame(base_img, config=map_config, robot_xy=state_xy)
                writer.stdin.write(final_frame.astype(np.uint8).tobytes())
                break
        writer.stdin.close()
        code = writer.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed on rollout {ep_idx} with exit code {code}")
        success = bool(last_info.get("success", False)) and (
            not bool(last_info.get("branch_mismatch", False))
        )
        if success:
            success_count += 1
            task_success_counts[task_id] += 1
        if last_info.get("branch_mismatch", False):
            branch_failure_count += 1
            task_branch_failure_counts[task_id] += 1
        result = {
            "episode_index": ep_idx,
            "task_id": task_id,
            "task_code": task_spec.task_code,
            "start_region_name": str(last_info.get("start_region_name")),
            "target_goal_name": str(last_info.get("target_goal_name")),
            "video_path": str(video_path),
            "final_position": [float(state_xy[0]), float(state_xy[1])],
            "final_phase_name": str(last_info.get("phase_name")),
            "reached_goal": last_info.get("reached_goal"),
            "branch_mismatch": bool(last_info.get("branch_mismatch", False)),
            "branch_mismatch_stage": last_info.get("branch_mismatch_stage"),
            "expected_branch_phase": last_info.get("expected_branch_phase"),
            "observed_branch_phase": last_info.get("observed_branch_phase"),
            "failure_reason": last_info.get("failure_reason"),
            "success": success,
            "steps": int(len(trajectory) - 1),
            "sum_reward": float(episode_reward),
            "collision_mode": "reject",
            "collision_detections": int(episode_collision_detection_count),
            "collision_rejections": int(episode_collision_rejection_count),
        }
        memory_debug_stats = get_visual_memory_debug_stats(policy)
        if memory_debug_stats is not None:
            result["visual_memory_debug"] = memory_debug_stats
        results.append(result)
        memory_debug_text = ""
        if memory_debug_stats is not None:
            memory_debug_text = f" memory_updates={int(memory_debug_stats.get('update_count', 0))} memory_norm={float(memory_debug_stats.get('state_norm', 0.0)):.4f}"
        print(
            f"[{ep_idx + 1:03d}/{eval_num_rollouts:03d}] task={task_spec.task_code}->{task_spec.target_goal_name} success={success} steps={result['steps']} reached={result['reached_goal']} branch_mismatch={result['branch_mismatch']} video={video_path.name}{memory_debug_text}"
        )
    per_task = {
        str(task_id): {
            "goal_name": TASK_ID_TO_GOAL_NAME[task_id],
            "rollouts": int(task_rollout_counts[task_id]),
            "success_count": int(task_success_counts[task_id]),
            "wrong_branch_failures": int(task_branch_failure_counts[task_id]),
            "success_rate": float(
                task_success_counts[task_id] / max(1, task_rollout_counts[task_id])
            ),
        }
        for task_id in sorted(TASK_ID_TO_GOAL_NAME)
    }
    summary = {
        "env": ENV_NAME,
        "policy_type": policy_type,
        "num_rollouts": eval_num_rollouts,
        "success_count": success_count,
        "success_rate": float(success_count / max(1, eval_num_rollouts)),
        "wrong_branch_failures": branch_failure_count,
        "seed": eval_seed,
        "fps": eval_fps,
        "max_steps": eval_max_steps,
        "max_action_step": args.max_action_step,
        "collision_mode": "reject",
        "collision_detections": int(sum((result["collision_detections"] for result in results))),
        "collision_rejections": int(sum((result["collision_rejections"] for result in results))),
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
    print(
        f"Collision mode: {summary['collision_mode']}, detections={summary['collision_detections']}, rejections={summary['collision_rejections']}"
    )
    print(f"Wrong-branch failures: {branch_failure_count}/{eval_num_rollouts}")
    for task_id in sorted(TASK_ID_TO_GOAL_NAME):
        task_summary = per_task[str(task_id)]
        print(
            f"  task {task_id} ({task_summary['goal_name']}): {task_summary['success_count']}/{task_summary['rollouts']} = {task_summary['success_rate']:.3f}, wrong_branch={task_summary['wrong_branch_failures']}"
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
    px = int(np.clip(px, 0, width - 1))
    py = int(np.clip(py, 0, height - 1))
    return (px, py)


def _fill_region(image: np.ndarray, region, color: tuple[int, int, int], config: MapConfig) -> None:
    if hasattr(region, "rectangles"):
        for rectangle in region.rectangles:
            _fill_region(image, rectangle, color, config=config)
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


def make_lerobot_base_image(
    config: MapConfig, image_size: int = DEFAULT_VIDEO_IMAGE_SIZE
) -> np.ndarray:
    base_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    base_image[:, :] = np.asarray(MAP_BACKGROUND_COLOR, dtype=np.uint8)
    for obstacle in config.obstacle_rectangles:
        _fill_region(base_image, obstacle, OBSTACLE_COLOR, config=config)
    for start_region in config.task_start_regions:
        _fill_region(base_image, start_region, START_COLOR, config=config)
    for goal_region in config.goal_regions:
        _fill_region(
            base_image,
            goal_region,
            GOAL_COLOR_BY_NAME.get(goal_region.name, (0, 0, 0)),
            config=config,
        )
    return base_image


def render_lerobot_frame(
    base_image: np.ndarray, config: MapConfig, robot_xy: tuple[float, float]
) -> np.ndarray:
    frame = base_image.copy()
    height, width = frame.shape[:2]
    robot_px = _workspace_to_pixel(robot_xy, config=config, image_size=(height, width))
    outer_radius = max(3, int(min(height, width) * 0.03))
    inner_radius = max(2, int(outer_radius * 0.65))
    _draw_disk(frame, robot_px, outer_radius, ROBOT_OUTER_COLOR)
    _draw_disk(frame, robot_px, inner_radius, ROBOT_INNER_COLOR)
    return frame


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
