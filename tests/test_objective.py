from __future__ import annotations

import numpy as np

from src.environment import Environment
from src.objective import (
    compute_phi,
    global_score,
    local_conflict_contribution,
    select_target_agent,
    sum_of_costs,
)


def test_sum_of_costs_single_and_multi_agent_shapes() -> None:
    single = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=float,
    )
    multi = np.array(
        [
            single,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 3.0],
                ],
                dtype=float,
            ),
        ],
        dtype=float,
    )

    assert np.isclose(sum_of_costs(single), 10.0)
    assert np.isclose(sum_of_costs(multi), 13.0)


def test_compute_phi_zero_for_well_separated_paths() -> None:
    env = Environment.build("Narrow")
    radius = 0.2
    dt = 0.1

    paths = np.array(
        [
            [[1.0, 2.4], [2.0, 2.4], [3.0, 2.4]],
            [[1.0, 0.6], [2.0, 0.6], [3.0, 0.6]],
        ],
        dtype=float,
    )

    phi = compute_phi(paths, env, radius, dt)
    assert phi >= 0.0
    assert np.isclose(phi, 0.0)


def test_compute_phi_penalizes_waypoint_agent_overlap() -> None:
    env = Environment.build("Narrow")
    radius = 0.3
    dt = 0.2

    paths = np.array(
        [
            [[2.0, 2.5], [3.0, 2.5], [4.0, 2.5]],
            [[2.0, 1.5], [3.0, 2.5], [4.0, 1.5]],  # same waypoint at k=1
        ],
        dtype=float,
    )

    phi = compute_phi(paths, env, radius, dt)
    assert phi > 0.0


def test_compute_phi_penalizes_swept_crossing_even_when_waypoints_are_separate() -> None:
    env = Environment.build("Narrow")
    radius = 0.25
    dt = 0.1

    # At the endpoints they are separated, but the segments cross in the middle.
    paths = np.array(
        [
            [[2.0, 2.0], [4.0, 3.0]],
            [[2.0, 3.0], [4.0, 2.0]],
        ],
        dtype=float,
    )

    phi = compute_phi(paths, env, radius, dt)
    assert phi > 0.0


def test_compute_phi_penalizes_obstacle_proximity_at_waypoints() -> None:
    env = Environment.build("Narrow")
    radius = 0.3
    dt = 0.1

    # Near the obstacle face x=5.0, but not inside.
    near_obstacle = np.array(
        [
            [[4.72, 4.2], [4.72, 4.2], [4.72, 4.2]],
        ],
        dtype=float,
    )

    far_from_obstacle = np.array(
        [
            [[2.0, 2.5], [2.0, 2.5], [2.0, 2.5]],
        ],
        dtype=float,
    )

    assert compute_phi(near_obstacle, env, radius, dt) > compute_phi(far_from_obstacle, env, radius, dt)


def test_compute_phi_penalizes_obstacle_proximity_on_swept_segment() -> None:
    env = Environment.build("Narrow")
    radius = 0.3
    dt = 0.1

    # Horizontal segment passes close to the left face of the upper-left obstacle.
    near_swept = np.array(
        [
            [[4.72, 4.0], [4.72, 4.3]],
        ],
        dtype=float,
    )

    far_swept = np.array(
        [
            [[2.0, 2.0], [2.0, 2.3]],
        ],
        dtype=float,
    )

    assert compute_phi(near_swept, env, radius, dt) > compute_phi(far_swept, env, radius, dt)


def test_compute_phi_penalizes_boundary_proximity_via_obstacle_term_extension() -> None:
    env = Environment.build("Narrow")
    radius = 0.3
    dt = 0.1

    near_wall = np.array(
        [
            [[0.32, 2.5], [0.32, 2.5], [0.32, 2.5]],
        ],
        dtype=float,
    )
    centered = np.array(
        [
            [[2.0, 2.5], [2.0, 2.5], [2.0, 2.5]],
        ],
        dtype=float,
    )

    # Requires the objective/environment stack from the final repaired version.
    assert compute_phi(near_wall, env, radius, dt) > compute_phi(centered, env, radius, dt)


def test_local_conflict_contribution_detects_obstacle_heavy_agent() -> None:
    env = Environment.build("Narrow")
    radius = 0.3
    dt = 0.1

    paths = np.array(
        [
            [[4.72, 4.1], [4.72, 4.2], [4.72, 4.3]],  # near obstacle
            [[2.0, 2.5], [2.1, 2.5], [2.2, 2.5]],
        ],
        dtype=float,
    )

    c0 = local_conflict_contribution(paths, 0, env, radius, dt)
    c1 = local_conflict_contribution(paths, 1, env, radius, dt)

    assert c0 > c1


def test_select_target_agent_prefers_obstacle_heavy_agent_even_without_pair_conflict() -> None:
    env = Environment.build("Narrow")
    radius = 0.3
    dt = 0.1

    paths = np.array(
        [
            [[4.72, 4.1], [4.72, 4.2], [4.72, 4.3]],  # near obstacle only
            [[1.0, 0.6], [1.2, 0.6], [1.4, 0.6]],
            [[10.0, 2.5], [10.2, 2.5], [10.4, 2.5]],
        ],
        dtype=float,
    )

    target = select_target_agent(paths, env, radius, dt)
    assert target == 0


def test_global_score_wraps_phi_and_soc_consistently() -> None:
    env = Environment.build("Narrow")
    radius = 0.3
    dt = 0.1

    paths = np.array(
        [
            [[2.0, 2.5], [3.0, 2.5], [4.0, 2.5]],
            [[2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
        ],
        dtype=float,
    )

    score = global_score(paths, env, radius, dt)
    assert np.isclose(score.phi, compute_phi(paths, env, radius, dt))
    assert np.isclose(score.soc, sum_of_costs(paths))