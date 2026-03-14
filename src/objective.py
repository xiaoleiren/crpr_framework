from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.environment import Environment
from src.geometry import (
    point_to_rect_distance,
    segment_to_rect_distance,
    segment_to_segment_distance,
)


@dataclass(frozen=True)
class LexicographicScore:
    phi: float
    soc: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.phi, self.soc)


def sum_of_costs(paths: np.ndarray) -> float:
    if paths.ndim == 2:
        return float(np.sum(np.linalg.norm(np.diff(paths, axis=0), axis=1)))
    return float(np.sum(np.linalg.norm(np.diff(paths, axis=1), axis=2)))


def _pair_penalty_at_waypoints(paths: np.ndarray, radius: float, dt: float) -> float:
    n_agents, n_steps, _ = paths.shape
    penalty = 0.0
    for k in range(n_steps):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = float(np.linalg.norm(paths[i, k] - paths[j, k]))
                penalty += max(0.0, 2.0 * radius - dist) * dt
    return float(penalty)


def _pair_penalty_swept(paths: np.ndarray, radius: float, dt: float) -> float:
    n_agents, n_steps, _ = paths.shape
    penalty = 0.0
    for k in range(n_steps - 1):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = segment_to_segment_distance(
                    paths[i, k],
                    paths[i, k + 1],
                    paths[j, k],
                    paths[j, k + 1],
                )
                penalty += max(0.0, 2.0 * radius - dist) * dt
    return float(penalty)


def _boundary_penalty_for_path(path: np.ndarray, env: Environment, radius: float, dt: float) -> float:
    penalty = 0.0
    for k in range(path.shape[0]):
        raw_clearance = env.boundary_clearance(path[k])
        body_clearance = raw_clearance - radius
        penalty += max(0.0, radius - body_clearance) * dt
    return float(penalty)


def _obstacle_penalty_for_path(path: np.ndarray, env: Environment, radius: float, dt: float) -> float:
    penalty = 0.0
    n_steps = path.shape[0]

    if env.obstacles:
        for k in range(n_steps):
            point = path[k]
            min_point_dist = min(point_to_rect_distance(point, rect) for rect in env.obstacles)
            penalty += max(0.0, radius - min_point_dist) * dt

        for k in range(n_steps - 1):
            a = path[k]
            b = path[k + 1]
            min_swept_dist = min(segment_to_rect_distance(a, b, rect) for rect in env.obstacles)
            penalty += max(0.0, radius - min_swept_dist) * dt

    penalty += _boundary_penalty_for_path(path, env, radius, dt)
    return float(penalty)


def compute_phi(paths: np.ndarray, env: Environment, radius: float, dt: float) -> float:
    phi = 0.0

    phi += _pair_penalty_at_waypoints(paths, radius, dt)
    phi += _pair_penalty_swept(paths, radius, dt)

    for i in range(paths.shape[0]):
        phi += _obstacle_penalty_for_path(paths[i], env, radius, dt)

    return float(phi)


def local_conflict_contribution(
    paths: np.ndarray,
    target: int,
    env: Environment,
    radius: float,
    dt: float,
) -> float:
    n_agents, n_steps, _ = paths.shape
    contribution = 0.0

    for k in range(n_steps):
        for j in range(n_agents):
            if j == target:
                continue
            dist = float(np.linalg.norm(paths[target, k] - paths[j, k]))
            contribution += max(0.0, 2.0 * radius - dist) * dt

    for k in range(n_steps - 1):
        for j in range(n_agents):
            if j == target:
                continue
            dist = segment_to_segment_distance(
                paths[target, k],
                paths[target, k + 1],
                paths[j, k],
                paths[j, k + 1],
            )
            contribution += max(0.0, 2.0 * radius - dist) * dt

    contribution += _obstacle_penalty_for_path(paths[target], env, radius, dt)
    return float(contribution)


def select_target_agent(paths: np.ndarray, env: Environment, radius: float, dt: float) -> int:
    contributions = [
        local_conflict_contribution(paths, i, env, radius, dt)
        for i in range(paths.shape[0])
    ]
    return int(np.argmax(np.asarray(contributions, dtype=float)))


def global_score(paths: np.ndarray, env: Environment, radius: float, dt: float) -> LexicographicScore:
    return LexicographicScore(phi=compute_phi(paths, env, radius, dt), soc=sum_of_costs(paths))


def local_score(
    candidate: np.ndarray,
    paths: np.ndarray,
    target: int,
    env: Environment,
    radius: float,
    dt: float,
) -> LexicographicScore:
    trial = paths.copy()
    trial[target] = candidate
    return LexicographicScore(phi=compute_phi(trial, env, radius, dt), soc=sum_of_costs(candidate))


def lexicographically_better(left: LexicographicScore, right: LexicographicScore, tol: float = 1e-12) -> bool:
    if left.phi < right.phi - tol:
        return True
    if abs(left.phi - right.phi) <= tol and left.soc < right.soc - tol:
        return True
    return False