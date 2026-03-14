from __future__ import annotations

import numpy as np
import pytest

from src.environment import Environment
from src.rrt_star import RRTStar


def test_rrt_fallback_does_not_return_known_colliding_segment() -> None:
    env = Environment.build("Office")
    planner = RRTStar(
        env=env,
        start=np.array([5.0, 6.0]),
        goal=np.array([7.0, 6.0]),
        agent_radius=0.3,
        max_iter=0,
        step_size=0.75,
        goal_bias=0.1,
        neighbor_radius=1.5,
        rng=np.random.default_rng(0),
    )

    with pytest.raises(RuntimeError):
        planner.plan()