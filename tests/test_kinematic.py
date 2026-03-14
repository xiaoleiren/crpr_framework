from __future__ import annotations

import numpy as np

from src.config import CRPRConfig
from src.kinematic import bidirectional_kinematic_projection, kinematics_within_limits


def test_bidirectional_projection_enforces_velocity_and_acceleration_limits() -> None:
    cfg = CRPRConfig.from_mapping(
        {
            "delta_alpha": 0.1,
            "beta": 0.5,
            "K": 2,
            "elite_max": 4,
            "d_max": 0.05,
            "epsilon": 0.001,
            "v_max": 1.0,
            "a_max": 2.0,
            "r_agent": 0.3,
            "dt": 0.05,
            "timeout_s": 0.5,
            "rrt_max_iter": 150,
            "rrt_step_size": 0.8,
            "rrt_goal_bias": 0.2,
            "rrt_neighbor_radius": 1.5,
            "min_clearance_skip": 0.02,
            "init_noise_sigma": 0.02,
            "cache_refresh_every": 2,
            "ablation": "full",
            "seed": 0,
        }
    )

    x = np.linspace(0.0, 1.0, 21)
    path = np.stack([x, np.sin(np.linspace(0.0, 4.0 * np.pi, 21)) * 5.0], axis=1)

    projected = bidirectional_kinematic_projection(
        path,
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        cfg,
    )

    assert np.allclose(projected[0], [0.0, 0.0])
    assert np.allclose(projected[-1], [1.0, 0.0])
    assert kinematics_within_limits(projected, cfg)