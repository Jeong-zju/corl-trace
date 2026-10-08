from __future__ import annotations
from typing import Any
import cv2
import numpy as np


def _array(value: Any, *, dtype=None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else value + 1


def _fit_bgr_to_canvas(
    frame: np.ndarray, *, width: int, height: int, pad_color: tuple[int, int, int] = (0, 0, 0)
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((height, width, 3), pad_color, dtype=np.uint8)
    src_h, src_w = (int(frame.shape[0]), int(frame.shape[1]))
    if src_h <= 0 or src_w <= 0:
        return (canvas, (0, 0, width, height))
    scale = min(width / float(src_w), height / float(src_h))
    fit_w = max(1, min(width, int(round(src_w * scale))))
    fit_h = max(1, min(height, int(round(src_h * scale))))
    x0 = (width - fit_w) // 2
    y0 = (height - fit_h) // 2
    resized = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_LINEAR)
    canvas[y0 : y0 + fit_h, x0 : x0 + fit_w] = resized
    return (canvas, (x0, y0, fit_w, fit_h))


def _image_to_bgr(
    image: np.ndarray | None,
    *,
    color_order: str,
    width: int,
    height: int,
    preserve_aspect: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image is None:
        return (np.zeros((height, width, 3), dtype=np.uint8), (0, 0, width, height))
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return (np.zeros((height, width, 3), dtype=np.uint8), (0, 0, width, height))
    if color_order == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if preserve_aspect:
        frame, rect = _fit_bgr_to_canvas(frame, width=width, height=height)
        return (np.ascontiguousarray(frame), rect)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return (np.ascontiguousarray(frame), (0, 0, width, height))


def _draw_label(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.5) -> None:
    cv2.putText(
        image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )


def _attention_lift_heatmap(
    values: np.ndarray, *, uniform_token_mass: float, red_lift: float = 3.0
) -> np.ndarray:
    heatmap = np.asarray(values, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    if heatmap.size == 0 or uniform_token_mass <= 1e-12:
        return np.zeros_like(heatmap, dtype=np.uint8)
    lift = heatmap / float(uniform_token_mass)
    normalized = (lift - 1.0) / max(1e-06, red_lift - 1.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _edge_mass_ratio(values: np.ndarray) -> float:
    heatmap = np.asarray(values, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0:
        return 0.0
    total = float(np.nan_to_num(heatmap, nan=0.0).sum())
    if total <= 1e-12:
        return 0.0
    edge_mask = np.zeros(heatmap.shape, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    return float(heatmap[edge_mask].sum() / total)


def _draw_bar_row(
    canvas: np.ndarray,
    *,
    title: str,
    values: np.ndarray | None,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    symmetric: bool = False,
) -> None:
    cv2.putText(
        canvas, title, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (35, 35, 35), 1, cv2.LINE_AA
    )
    if values is None or values.size == 0:
        cv2.putText(
            canvas,
            "n/a",
            (x + 130, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        return
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(values.shape[0])
    gap = 5
    slot_w = max(1, int((width - gap * max(0, n - 1)) / max(1, n)))
    if symmetric:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        denom = max(max_abs, 1e-06)
        center_y = y + height // 2
        cv2.line(canvas, (x, center_y), (x + width, center_y), (180, 180, 180), 1)
        for idx, value in enumerate(values):
            bar_h = int(abs(float(value)) / denom * (height // 2 - 2))
            x0 = x + idx * (slot_w + gap)
            if value >= 0:
                y0, y1 = (center_y - bar_h, center_y)
                bar_color = color
            else:
                y0, y1 = (center_y, center_y + bar_h)
                bar_color = (80, 80, 210)
            cv2.rectangle(canvas, (x0, y0), (x0 + slot_w, y1), bar_color, -1)
            cv2.putText(
                canvas,
                str(idx),
                (x0 + 2, y + height + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
        return
    vmax = float(np.max(values)) if values.size else 0.0
    denom = max(vmax, 1e-06)
    for idx, value in enumerate(values):
        bar_h = int(max(float(value), 0.0) / denom * (height - 2))
        x0 = x + idx * (slot_w + gap)
        y0 = y + height - bar_h
        cv2.rectangle(canvas, (x0, y), (x0 + slot_w, y + height), (230, 230, 230), 1)
        cv2.rectangle(canvas, (x0 + 1, y0), (x0 + slot_w - 1, y + height - 1), color, -1)
        cv2.putText(
            canvas,
            str(idx),
            (x0 + 2, y + height + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )


def _slot_vector(value: Any, *, num_slots: int | None = None) -> np.ndarray | None:
    arr = _array(value, dtype=np.float32)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2:
        if num_slots is not None and arr.shape[0] == num_slots:
            return arr.reshape(num_slots, -1).mean(axis=1)
        return arr[0].reshape(-1)
    if arr.ndim >= 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        if num_slots is not None and arr.shape[0] == num_slots:
            return arr.reshape(num_slots, -1).mean(axis=1)
        return arr.reshape(arr.shape[0], -1).mean(axis=1)
    return arr.reshape(-1)


def make_slot_memory_history_sample(debug: dict[str, Any] | None) -> dict[str, Any]:
    """Record the actual memory trajectory for an evaluation episode."""
    slot = dict((debug or {}).get("slot_memory") or {})
    return {
        key: np.asarray(slot[key], dtype=np.float32).copy()
        for key in (
            "routing_weights",
            "readout_weights",
            "write_strength",
            "memory_next_norm",
            "memory_delta_norm",
        )
        if slot.get(key) is not None
    }


def render_attention_panel(
    *,
    images: dict[str, np.ndarray],
    debug: dict[str, Any] | None,
    color_order: str,
    camera_labels: dict[str, str],
    overlay_alpha: float,
    query_step: int,
    tile_width: int = 300,
    tile_height: int = 300,
) -> np.ndarray:
    debug = debug or {}
    layout = dict(debug.get("decoder_token_layout") or {})
    attn = _array(debug.get("decoder_cross_attention"), dtype=np.float32)
    camera_keys = list(layout.get("camera_keys") or images.keys())
    num_cameras = int(layout.get("num_cameras", len(camera_keys)) or len(camera_keys))
    feature_h = int(layout.get("feature_h", 0) or 0)
    feature_w = int(layout.get("feature_w", 0) or 0)
    image_token_start = int(layout.get("image_token_start", 0) or 0)
    image_token_count = int(layout.get("image_token_count", 0) or 0)
    query_step = max(0, int(query_step))
    maps: list[np.ndarray | None] = [None for _ in range(max(num_cameras, len(camera_keys)))]
    token_attention_sum = 0.0
    uniform_image_token_mass = 0.0
    if (
        attn is not None
        and attn.ndim == 2
        and (feature_h > 0)
        and (feature_w > 0)
        and (num_cameras > 0)
    ):
        query_idx = min(query_step, attn.shape[0] - 1)
        source = attn[query_idx]
        end = image_token_start + image_token_count
        if image_token_count == num_cameras * feature_h * feature_w and end <= source.shape[0]:
            image_attention = source[image_token_start:end].reshape(
                num_cameras, feature_h, feature_w
            )
            token_attention_sum = float(image_attention.sum())
            uniform_image_token_mass = token_attention_sum / float(max(1, image_token_count))
            maps = [image_attention[idx] for idx in range(num_cameras)]
    width = max(1, len(camera_keys)) * tile_width
    extra_labels = list(layout.get("extra_memory_token_labels") or [])
    extra_label_cols = max(1, width // 158)
    extra_label_rows = 0 if not extra_labels else (len(extra_labels) - 1) // extra_label_cols + 1
    header_h = 44
    footer_h = 82 + max(0, extra_label_rows - 1) * 38
    canvas = np.full((header_h + tile_height + footer_h, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"decoder cross-attention overlay | query_step={query_step}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    for idx, camera_key in enumerate(camera_keys):
        x0 = idx * tile_width
        frame, content_rect = _image_to_bgr(
            images.get(camera_key), color_order=color_order, width=tile_width, height=tile_height
        )
        heat = maps[idx] if idx < len(maps) else None
        if heat is not None:
            heat_u8 = _attention_lift_heatmap(heat, uniform_token_mass=uniform_image_token_mass)
            rect_x, rect_y, rect_w, rect_h = content_rect
            heat_u8 = cv2.resize(heat_u8, (rect_w, rect_h), interpolation=cv2.INTER_LINEAR)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            content = frame[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w]
            frame[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w] = cv2.addWeighted(
                content, 1.0 - overlay_alpha, heat_color, overlay_alpha, 0.0
            )
            heat_sum = float(np.asarray(heat, dtype=np.float32).sum())
            peak_lift = (
                float(np.asarray(heat, dtype=np.float32).max() / uniform_image_token_mass)
                if uniform_image_token_mass > 1e-12
                else 0.0
            )
            edge_mass = _edge_mass_ratio(heat)
        else:
            heat_sum = 0.0
            peak_lift = 0.0
            edge_mass = 0.0
        label = camera_labels.get(camera_key, camera_key.rsplit(".", 1)[-1])
        _draw_label(
            frame, f"{label} sum={heat_sum:.3f} peak={peak_lift:.1f}x edge={edge_mass:.0%}", 8, 22
        )
        canvas[header_h : header_h + tile_height, x0 : x0 + tile_width] = frame
    footer_y = header_h + tile_height + 24
    cv2.putText(
        canvas,
        f"image-token attention total={token_attention_sum:.3f} | overlay shows >uniform lift, red ~= >=3x uniform",
        (12, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    if attn is not None and attn.ndim == 2 and extra_labels:
        query_idx = min(query_step, attn.shape[0] - 1)
        values = attn[query_idx, : len(extra_labels)]
        max_v = max(float(np.max(values)), 1e-06)
        for idx, (label, value) in enumerate(zip(extra_labels, values, strict=False)):
            col = idx % extra_label_cols
            row = idx // extra_label_cols
            x = 12 + col * 158
            y = footer_y + 22 + row * 38
            bar_w = int(130 * float(value) / max_v)
            cv2.rectangle(canvas, (x, y - 12), (x + 130, y + 2), (222, 222, 222), 1)
            cv2.rectangle(canvas, (x, y - 11), (x + bar_w, y + 1), (70, 145, 210), -1)
            cv2.putText(
                canvas,
                f"{label}:{float(value):.3f}",
                (x, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (55, 55, 55),
                1,
                cv2.LINE_AA,
            )
    return canvas


def render_slot_memory_panel(
    *,
    debug: dict[str, Any] | None,
    images: dict[str, np.ndarray] | None = None,
    color_order: str = "rgb",
    camera_labels: dict[str, str] | None = None,
    history: list[dict[str, Any]] | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
    width: int = 760,
    height: int = 560,
) -> np.ndarray:
    """Display current slot read/write activity and recorded memory norms."""
    width, height = (_even(max(320, width)), _even(max(360, height)))
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    label = "TRACE slot memory"
    if current_step is not None:
        label += f" | step {current_step}"
    _draw_label(canvas, label, 14, 28, 0.6)
    slot = dict((debug or {}).get("slot_memory") or {})
    camera_labels = camera_labels or {}
    image_height = 0
    if images:
        image_height = min(150, height // 4)
        tile_width = max(1, width // len(images))
        for index, (key, value) in enumerate(images.items()):
            tile, _ = _image_to_bgr(
                value, color_order=color_order, width=tile_width, height=image_height
            )
            _draw_label(tile, camera_labels.get(key, key.rsplit(".", 1)[-1]), 6, 18)
            start = index * tile_width
            canvas[38 : 38 + image_height, start : start + tile_width] = tile
    rows = (
        ("routing_weights", "Routing"),
        ("readout_weights", "Readout"),
        ("write_strength", "Write strength"),
        ("memory_next_norm", "Memory norm"),
    )
    top = 65 + image_height
    row_height = max(44, (height - top - 35) // len(rows))
    for index, (key, title) in enumerate(rows):
        values = _slot_vector(slot.get(key))
        _draw_bar_row(
            canvas,
            title=title,
            values=values,
            x=16,
            y=top + index * row_height,
            width=width - 32,
            height=max(22, row_height - 22),
            color=(180, 115, 45),
        )
    _draw_label(canvas, f"Recorded frames: {len(history or [])}", 16, height - 12, 0.4)
    return canvas


def _stats_vector(stats: dict[str, Any] | None, key: str) -> np.ndarray | None:
    if not isinstance(stats, dict):
        return None
    value = stats.get(key)
    arr = _array(value, dtype=np.float32)
    if arr is None or arr.size == 0:
        return None
    return arr.reshape(-1)


def render_signature_panel(
    *, signature_debug: dict[str, Any] | None, width: int = 920, height: int = 540
) -> np.ndarray:
    signature_debug = signature_debug or {}
    signature = _array(signature_debug.get("signature"), dtype=np.float32)
    delta = _array(signature_debug.get("delta_signature"), dtype=np.float32)
    stats = signature_debug.get("signature_stats")
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        canvas,
        "path signature vs dataset distribution",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"backend={signature_debug.get('backend', 'unknown')} window={signature_debug.get('window_length', 0)}/{signature_debug.get('history_length', 'full')} dataset={signature_debug.get('dataset_root') or 'unknown'}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    if signature is None:
        cv2.putText(
            canvas,
            "No path signature has been computed yet.",
            (18, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (95, 95, 95),
            1,
            cv2.LINE_AA,
        )
        return canvas
    signature = signature.reshape(-1)
    delta = np.zeros_like(signature) if delta is None else delta.reshape(-1)
    mean = _stats_vector(stats, "mean")
    std = _stats_vector(stats, "std")
    q10 = _stats_vector(stats, "q10")
    q90 = _stats_vector(stats, "q90")
    vmin = _stats_vector(stats, "min")
    vmax = _stats_vector(stats, "max")
    sig_norm = float(np.linalg.norm(signature))
    delta_norm = float(np.linalg.norm(delta))
    cv2.putText(
        canvas,
        f"dim={signature.size} | ||sig||={sig_norm:.3f} | ||delta||={delta_norm:.3f}",
        (18, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    if mean is None or std is None or mean.size != signature.size:
        cv2.putText(
            canvas,
            "Dataset signature stats are missing. Run dataset signature-stat generation or provide meta/stats.json entries for observation.path_signature.",
            (18, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (55, 55, 160),
            1,
            cv2.LINE_AA,
        )
        sample = signature[: min(96, signature.size)]
        mean = np.zeros_like(signature)
        std = np.ones_like(signature)
        selected = np.arange(sample.size)
    else:
        z = (signature - mean) / np.maximum(std, 1e-06)
        selected = np.argsort(-np.abs(z))[: min(96, signature.size)]
        out_of_range = 0.0
        if vmin is not None and vmax is not None and (vmin.size == signature.size):
            out_of_range = float(np.mean((signature < vmin) | (signature > vmax)))
        cv2.putText(
            canvas,
            f"z_abs_mean={float(np.mean(np.abs(z))):.3f} z_abs_max={float(np.max(np.abs(z))):.3f} out_of_minmax={out_of_range * 100:.1f}%",
            (18, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
    plot_x, plot_y = (48, 170)
    plot_w, plot_h = (width - 90, height - 210)
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (210, 210, 210), 1)
    current = signature[selected]
    center = (
        mean[selected]
        if mean is not None and mean.size == signature.size
        else np.zeros_like(current)
    )
    spread = (
        std[selected] if std is not None and std.size == signature.size else np.ones_like(current)
    )
    low = q10[selected] if q10 is not None and q10.size == signature.size else center - spread
    high = q90[selected] if q90 is not None and q90.size == signature.size else center + spread
    y_values = np.concatenate([current, low, high, center])
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if y_max <= y_min + 1e-09:
        y_max = y_min + 1.0

    def xy(idx: int, value: float) -> tuple[int, int]:
        x = plot_x + int(idx / max(1, len(selected) - 1) * plot_w)
        y = plot_y + plot_h - int((float(value) - y_min) / (y_max - y_min) * plot_h)
        return (x, y)

    for idx in range(len(selected) - 1):
        cv2.line(canvas, xy(idx, low[idx]), xy(idx + 1, low[idx + 1]), (185, 205, 235), 1)
        cv2.line(canvas, xy(idx, high[idx]), xy(idx + 1, high[idx + 1]), (185, 205, 235), 1)
        cv2.line(canvas, xy(idx, center[idx]), xy(idx + 1, center[idx + 1]), (150, 150, 150), 1)
        cv2.line(canvas, xy(idx, current[idx]), xy(idx + 1, current[idx + 1]), (30, 90, 220), 2)
    cv2.putText(
        canvas,
        "blue=current | gray=dataset mean | pale band=q10/q90 or +/-1std",
        (plot_x, plot_y + plot_h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas
