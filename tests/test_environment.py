from __future__ import annotations

import numpy as np

from src.environment import Environment


def test_sampled_points_are_in_free_space() -> None:
    env = Environment.build("Narrow")
    starts, goals = env.sample_start_goal(n=4, radius=0.3, seed=7)
    for point in np.vstack([starts, goals]):
        assert env.in_bounds(point, 0.3)
        assert not env.collides_with_obstacle(point, 0.3)


def test_project_to_bounds_clips_with_margin() -> None:
    env = Environment.build("Narrow")
    point = np.array([-1.0, 9.0], dtype=float)
    clipped = env.project_to_bounds(point, margin=0.3)

    assert np.allclose(clipped, [0.3, env.bounds[1] - 0.3])
    assert env.in_bounds(clipped, 0.3)


def test_boundary_clearance_matches_distance_to_nearest_wall() -> None:
    env = Environment.build("Narrow")
    point = np.array([2.0, 1.5], dtype=float)
    clearance = env.boundary_clearance(point)

    assert clearance == min(
        point[0],
        env.bounds[0] - point[0],
        point[1],
        env.bounds[1] - point[1],
    )