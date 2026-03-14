from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def small_cfg_dict() -> dict:
    return {
        "delta_alpha": 0.1,
        "beta": 0.5,
        "K": 3,
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