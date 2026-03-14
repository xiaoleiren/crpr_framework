from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Rectangle:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, point: np.ndarray) -> bool:
        return bool(
            self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax
        )

    def nearest_point(self, point: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(point[0], self.xmin, self.xmax),
                np.clip(point[1], self.ymin, self.ymax),
            ],
            dtype=float,
        )


def pairwise_sup_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(path_a - path_b, axis=1)))


def point_to_rect_distance(point: np.ndarray, rect: Rectangle) -> float:
    nearest = rect.nearest_point(point)
    return float(np.linalg.norm(point - nearest))


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool(
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)

    if (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4):
        return True
    if abs(o1) < 1e-9 and on_segment(a1, b1, a2):
        return True
    if abs(o2) < 1e-9 and on_segment(a1, b2, a2):
        return True
    if abs(o3) < 1e-9 and on_segment(b1, a1, b2):
        return True
    if abs(o4) < 1e-9 and on_segment(b1, a2, b2):
        return True
    return False


def point_to_segment_distance(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    ab = seg_b - seg_a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - seg_a))
    t = float(np.clip(np.dot(point - seg_a, ab) / denom, 0.0, 1.0))
    proj = seg_a + t * ab
    return float(np.linalg.norm(point - proj))


def segment_to_segment_distance(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> float:
    if segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        point_to_segment_distance(a1, b1, b2),
        point_to_segment_distance(a2, b1, b2),
        point_to_segment_distance(b1, a1, a2),
        point_to_segment_distance(b2, a1, a2),
    )


def segment_to_rect_distance(seg_a: np.ndarray, seg_b: np.ndarray, rect: Rectangle) -> float:
    if rect.contains(seg_a) or rect.contains(seg_b):
        return 0.0
    corners = [
        np.array([rect.xmin, rect.ymin], dtype=float),
        np.array([rect.xmax, rect.ymin], dtype=float),
        np.array([rect.xmax, rect.ymax], dtype=float),
        np.array([rect.xmin, rect.ymax], dtype=float),
    ]
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    if any(segments_intersect(seg_a, seg_b, e0, e1) for e0, e1 in edges):
        return 0.0
    distances = [point_to_rect_distance(seg_a, rect), point_to_rect_distance(seg_b, rect)]
    distances.extend(segment_to_segment_distance(seg_a, seg_b, e0, e1) for e0, e1 in edges)
    return float(min(distances))


def polyline_length(polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))


def sample_polyline_by_arclength(polyline: np.ndarray, distances: np.ndarray) -> np.ndarray:
    if len(polyline) == 1:
        return np.repeat(polyline, len(distances), axis=0)
    seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total <= 1e-12:
        return np.repeat(polyline[[0]], len(distances), axis=0)
    distances = np.clip(distances, 0.0, total)
    samples = np.zeros((len(distances), 2), dtype=float)
    for idx, s in enumerate(distances):
        seg_idx = int(np.searchsorted(cumulative, s, side="right") - 1)
        seg_idx = min(seg_idx, len(seg_lengths) - 1)
        local_len = seg_lengths[seg_idx]
        if local_len <= 1e-12:
            samples[idx] = polyline[seg_idx]
            continue
        alpha = (s - cumulative[seg_idx]) / local_len
        samples[idx] = (1.0 - alpha) * polyline[seg_idx] + alpha * polyline[seg_idx + 1]
    return samples


def time_align_paths(polylines: Sequence[np.ndarray], dt: float, nominal_speed: float) -> Tuple[np.ndarray, np.ndarray, float]:
    durations: List[float] = []
    lengths: List[float] = []
    for polyline in polylines:
        length = polyline_length(polyline)
        lengths.append(length)
        durations.append(max(length / max(nominal_speed, 1e-9), dt))
    t_max = max(durations)
    m = int(np.ceil(t_max / dt)) + 1
    aligned = np.zeros((len(polylines), m, 2), dtype=float)
    arrival_times = np.array(durations, dtype=float)
    time_grid = np.arange(m, dtype=float) * dt
    for i, polyline in enumerate(polylines):
        length = lengths[i]
        duration = durations[i]
        if length <= 1e-12:
            aligned[i] = polyline[-1]
            continue
        active = time_grid <= duration + 1e-12
        active_times = np.clip(time_grid[active], 0.0, duration)
        distances = (active_times / max(duration, 1e-12)) * length
        aligned[i, active] = sample_polyline_by_arclength(polyline, distances)
        aligned[i, ~active] = polyline[-1]
    return aligned, arrival_times, t_max


def compute_makespan(aligned_paths: np.ndarray, goals: np.ndarray, dt: float, atol: float = 1e-3) -> float:
    last_arrival = 0.0
    for i in range(aligned_paths.shape[0]):
        reached = np.linalg.norm(aligned_paths[i] - goals[i], axis=1) <= atol
        if np.any(reached):
            first_idx = int(np.argmax(reached))
            last_arrival = max(last_arrival, first_idx * dt)
        else:
            last_arrival = max(last_arrival, (aligned_paths.shape[1] - 1) * dt)
    return float(last_arrival)
