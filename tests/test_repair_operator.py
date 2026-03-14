from __future__ import annotations

import numpy as np

from src.collision import CollisionCache, swept_feasible
from src.config import CRPRConfig
from src.environment import Environment
from src.kinematic import kinematics_within_limits
from src.repair_operator import project_point_feasible, repair_l_beta


def test_project_point_feasible_pushes_point_back_inside_bounds() -> None:
    env = Environment.build("Narrow")
    point = np.array([-1.0, 10.0], dtype=float)

    repaired = project_point_feasible(
        point,
        env,
        radius=0.3,
        epsilon=1e-3,
    )

    assert env.in_bounds(repaired, 0.3)
    assert not env.collides_with_obstacle(repaired, 0.3)


def test_project_point_feasible_pushes_point_out_of_obstacle() -> None:
    env = Environment.build("Narrow")

    # inside left upper obstacle: x in [5,6], y in [3.1,5]
    point = np.array([5.5, 4.0], dtype=float)

    repaired = project_point_feasible(
        point,
        env,
        radius=0.3,
        epsilon=1e-3,
    )

    assert env.in_bounds(repaired, 0.3)
    assert not env.collides_with_obstacle(repaired, 0.3)


def test_repair_l_beta_preserves_start_goal_and_returns_bounded_path(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")

    current_paths = np.array(
        [
            [[2.0, 2.5], [3.0, 2.5], [4.0, 2.5], [5.0, 2.5]],
            [[5.0, 2.5], [4.0, 2.5], [3.0, 2.5], [2.0, 2.5]],
        ],
        dtype=float,
    )
    candidate = current_paths[0].copy()
    candidate[1] = np.array([0.0, 0.0], dtype=float)  # deliberately bad interior point
    goals = np.array(
        [
            current_paths[0, -1],
            current_paths[1, -1],
        ],
        dtype=float,
    )

    result = repair_l_beta(
        candidate=candidate,
        current_paths=current_paths,
        target=0,
        goals=goals,
        env=env,
        cfg=cfg,
        cache=CollisionCache(active_pairs={(0, 1)}, refresh_every=2),
    )

    assert result.trajectory is not None
    assert np.allclose(result.trajectory[0], current_paths[0, 0])
    assert np.allclose(result.trajectory[-1], goals[0])

    for point in result.trajectory:
        assert env.in_bounds(point, cfg.r_agent)

    assert kinematics_within_limits(result.trajectory, cfg)


def test_repair_l_beta_phi_history_length_matches_passes_used(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")

    current_paths = np.array(
        [
            [[2.0, 2.5], [3.0, 2.5], [4.0, 2.5]],
            [[4.0, 2.5], [3.0, 2.5], [2.0, 2.5]],
        ],
        dtype=float,
    )
    candidate = current_paths[0].copy()
    goals = np.array(
        [
            current_paths[0, -1],
            current_paths[1, -1],
        ],
        dtype=float,
    )

    result = repair_l_beta(
        candidate=candidate,
        current_paths=current_paths,
        target=0,
        goals=goals,
        env=env,
        cfg=cfg,
        cache=CollisionCache(active_pairs={(0, 1)}, refresh_every=2),
    )

    assert len(result.phi_history) == result.passes_used


def test_repair_output_is_consistent_with_swept_feasible_when_marked_feasible(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")

    current_paths = np.array(
        [
            [[2.0, 2.5], [3.0, 2.5], [4.0, 2.5], [5.0, 2.5]],
            [[5.0, 2.5], [4.0, 2.5], [3.0, 2.5], [2.0, 2.5]],
        ],
        dtype=float,
    )
    candidate = current_paths[0].copy()
    goals = np.array(
        [
            current_paths[0, -1],
            current_paths[1, -1],
        ],
        dtype=float,
    )

    cache = CollisionCache(active_pairs={(0, 1)}, refresh_every=2)
    result = repair_l_beta(
        candidate=candidate,
        current_paths=current_paths,
        target=0,
        goals=goals,
        env=env,
        cfg=cfg,
        cache=cache,
    )

    assert result.trajectory is not None

    if result.feasible:
        trial = current_paths.copy()
        trial[0] = result.trajectory
        feasible, _ = swept_feasible(trial, env, cfg.r_agent, cache)
        assert feasible