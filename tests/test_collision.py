from __future__ import annotations

import numpy as np

from src.collision import (
    CollisionCache,
    candidate_pairs_from_broadphase,
    swept_boundary_collision,
    swept_feasible,
)
from src.environment import Environment


def test_swept_boundary_collision_detects_out_of_bounds_waypoint() -> None:
    env = Environment.build("Narrow")
    radius = 0.3

    path = np.array(
        [
            [0.5, 0.5],
            [0.2, 0.5],   # violates margin=radius because x < 0.3
            [0.5, 0.5],
        ],
        dtype=float,
    )

    assert swept_boundary_collision(path, env, radius)


def test_swept_feasible_rejects_paths_that_violate_boundary_margin() -> None:
    env = Environment.build("Narrow")
    radius = 0.3

    paths = np.array(
        [
            [[0.5, 0.5], [0.2, 0.5], [0.5, 0.5]],
            [[2.0, 2.5], [2.2, 2.5], [2.4, 2.5]],
        ],
        dtype=float,
    )

    feasible, colliding_pairs = swept_feasible(paths, env, radius)
    assert not feasible
    assert colliding_pairs == set()


def test_swept_feasible_returns_colliding_agent_pair_when_paths_cross() -> None:
    env = Environment.build("Narrow")
    radius = 0.2

    paths = np.array(
        [
            [[2.0, 2.5], [4.0, 2.5]],
            [[4.0, 2.5], [2.0, 2.5]],
        ],
        dtype=float,
    )

    feasible, colliding_pairs = swept_feasible(paths, env, radius)
    assert not feasible
    assert colliding_pairs == {(0, 1)}


def test_candidate_pairs_from_broadphase_skips_far_apart_agents() -> None:
    radius = 0.3
    paths = np.array(
        [
            [[1.0, 1.0], [1.5, 1.0]],
            [[1.2, 1.0], [1.7, 1.0]],
            [[10.0, 4.0], [10.5, 4.0]],
        ],
        dtype=float,
    )

    pairs = candidate_pairs_from_broadphase(paths, radius)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs


def test_swept_feasible_uses_active_pairs_from_cache_when_not_refreshing() -> None:
    env = Environment.build("Narrow")
    radius = 0.2

    # pair (0,1) collides; agent 2 is far away
    paths = np.array(
        [
            [[2.0, 2.5], [4.0, 2.5]],
            [[4.0, 2.5], [2.0, 2.5]],
            [[10.0, 2.5], [11.0, 2.5]],
        ],
        dtype=float,
    )

    cache = CollisionCache(active_pairs={(0, 1)}, refresh_every=10, iteration=1)
    feasible, colliding_pairs = swept_feasible(paths, env, radius, cache)

    assert not feasible
    assert colliding_pairs == {(0, 1)}


def test_collision_cache_full_scan_when_active_pairs_empty() -> None:
    cache = CollisionCache(active_pairs=set(), refresh_every=10, iteration=7)
    assert cache.should_full_scan()


def test_collision_cache_step_advances_iteration() -> None:
    cache = CollisionCache(active_pairs=set(), refresh_every=3, iteration=0)
    cache.step()
    assert cache.iteration == 1