from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CRPRConfig:
    delta_alpha: float
    beta: float
    K: int
    elite_max: int
    d_max: float
    epsilon: float
    v_max: float
    a_max: float
    r_agent: float
    dt: float
    timeout_s: float
    rrt_max_iter: int
    rrt_step_size: float
    rrt_goal_bias: float
    rrt_neighbor_radius: float
    min_clearance_skip: float
    init_noise_sigma: float
    cache_refresh_every: int
    ablation: str = "full"
    seed: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CRPRConfig":
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (0.0 < self.delta_alpha <= 1.0):
            raise ValueError("delta_alpha must be in (0, 1].")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if self.K < 1:
            raise ValueError("K must be >= 1.")
        if self.elite_max < 2:
            raise ValueError("elite_max must be >= 2.")
        if self.d_max <= 0.0:
            raise ValueError("d_max must be positive.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.v_max <= 0.0 or self.a_max <= 0.0:
            raise ValueError("v_max and a_max must be positive.")
        if self.r_agent <= 0.0 or self.dt <= 0.0 or self.timeout_s <= 0.0:
            raise ValueError("r_agent, dt and timeout_s must be positive.")
        if self.rrt_max_iter < 1:
            raise ValueError("rrt_max_iter must be >= 1.")
        if self.rrt_step_size <= 0.0 or self.rrt_neighbor_radius <= 0.0:
            raise ValueError("RRT parameters must be positive.")
        if not (0.0 <= self.rrt_goal_bias <= 1.0):
            raise ValueError("rrt_goal_bias must be in [0, 1].")
        if self.min_clearance_skip < 0.0:
            raise ValueError("min_clearance_skip cannot be negative.")
        if self.init_noise_sigma < 0.0:
            raise ValueError("init_noise_sigma cannot be negative.")
        if self.cache_refresh_every < 1:
            raise ValueError("cache_refresh_every must be >= 1.")
        if self.ablation not in {"full", "no_relink", "no_repair"}:
            raise ValueError("ablation must be one of: full, no_relink, no_repair.")
