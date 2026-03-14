from __future__ import annotations

import numpy as np

from src.geometry import Rectangle, segment_to_rect_distance, segment_to_segment_distance, time_align_paths


def test_segment_to_segment_distance_zero_for_crossing_segments() -> None:
    a1 = np.array([0.0, 0.0])
    a2 = np.array([1.0, 1.0])
    b1 = np.array([0.0, 1.0])
    b2 = np.array([1.0, 0.0])
    assert segment_to_segment_distance(a1, a2, b1, b2) == 0.0


def test_segment_to_rect_distance_positive_outside_rectangle() -> None:
    rect = Rectangle(1.0, 1.0, 2.0, 2.0)
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    assert segment_to_rect_distance(a, b, rect) > 0.0


def test_time_align_paths_pads_earlier_goal() -> None:
    short = np.array([[0.0, 0.0], [1.0, 0.0]])
    long = np.array([[0.0, 0.0], [2.0, 0.0]])
    aligned, arrivals, tmax = time_align_paths([short, long], dt=0.5, nominal_speed=1.0)
    assert aligned.shape[0] == 2
    assert arrivals[0] < arrivals[1]
    assert np.allclose(aligned[0, -1], short[-1])
