from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.collision import CollisionCache, swept_feasible
from src.config import CRPRConfig
from src.environment import Environment
from src.kinematic import bidirectional_kinematic_projection, radial_clip
from src.objective import compute_phi


@dataclass
class RepairResult:
    trajectory: Optional[np.ndarray]
    feasible: bool
    passes_used: int
    phi_history: list[float]


def obstacle_repulsion(point: np.ndarray, env: Environment, epsilon: float) -> np.ndarray:
    grad = np.zeros(2, dtype=float)
    for rect in env.obstacles:
        closest = rect.nearest_point(point)
        direction = point - closest
        dist = float(np.linalg.norm(direction))
        if dist <= 1e-12 and rect.contains(point):
            center = np.array(
                [(rect.xmin + rect.xmax) * 0.5, (rect.ymin + rect.ymax) * 0.5],
                dtype=float,
            )
            direction = point - center
            if np.linalg.norm(direction) <= 1e-12:
                direction = np.array([1.0, 0.0], dtype=float)
            dist = float(np.linalg.norm(direction))
        grad += direction / (dist + epsilon)
    return grad


def boundary_repulsion(point: np.ndarray, env: Environment, epsilon: float) -> np.ndarray:
    x, y = float(point[0]), float(point[1])
    w, h = env.bounds

    # gradient of "stay away from walls"
    grad = np.zeros(2, dtype=float)
    grad[0] += 1.0 / (x + epsilon)
    grad[0] -= 1.0 / (w - x + epsilon)
    grad[1] += 1.0 / (y + epsilon)
    grad[1] -= 1.0 / (h - y + epsilon)
    return grad


def project_point_feasible(
    point: np.ndarray,
    env: Environment,
    radius: float,
    epsilon: float,
    max_iters: int = 12,
) -> np.ndarray:
    x = env.project_to_bounds(point, margin=radius)

    for _ in range(max_iters):
        if env.in_bounds(x, radius) and not env.collides_with_obstacle(x, radius):
            return x

        grad = np.zeros(2, dtype=float)
        grad += obstacle_repulsion(x, env, epsilon)
        grad += boundary_repulsion(x, env, epsilon)

        norm = float(np.linalg.norm(grad))
        if norm <= 1e-12:
            break

        step = min(radius * 0.5, 0.1)
        x = x + step * grad / norm
        x = env.project_to_bounds(x, margin=radius)

    return env.project_to_bounds(x, margin=radius)


def repair_l_beta(
    candidate: np.ndarray,
    current_paths: np.ndarray,
    target: int,
    goals: np.ndarray,
    env: Environment,
    cfg: CRPRConfig,
    cache: Optional[CollisionCache] = None,
) -> RepairResult:
    trajectory = candidate.copy()
    start = current_paths[target, 0].copy()
    goal = goals[target].copy()
    phi_history: list[float] = []

    trajectory[0] = start
    trajectory[-1] = goal
    for k in range(1, len(trajectory) - 1):
        trajectory[k] = project_point_feasible(
            trajectory[k],
            env,
            cfg.r_agent,
            cfg.epsilon,
        )

    for pass_idx in range(1, cfg.K + 1):
        trajectory = bidirectional_kinematic_projection(trajectory, start, goal, cfg)

        displaced = trajectory.copy()
        for k in range(1, len(displaced) - 1):
            grad = np.zeros(2, dtype=float)

            # repel from other agents at same aligned time index
            for j in range(current_paths.shape[0]):
                if j == target:
                    continue
                direction = displaced[k] - current_paths[j, k]
                dist = float(np.linalg.norm(direction))
                grad += direction / (dist + cfg.epsilon)

            # repel from obstacles and boundary
            grad += obstacle_repulsion(displaced[k], env, cfg.epsilon)
            grad += boundary_repulsion(displaced[k], env, cfg.epsilon)

            disp = cfg.beta * radial_clip(grad, cfg.d_max, cfg.epsilon)
            displaced[k] += disp
            displaced[k] = project_point_feasible(
                displaced[k],
                env,
                cfg.r_agent,
                cfg.epsilon,
            )

        displaced = bidirectional_kinematic_projection(displaced, start, goal, cfg)

        displaced[0] = start
        displaced[-1] = goal
        for k in range(1, len(displaced) - 1):
            displaced[k] = project_point_feasible(
                displaced[k],
                env,
                cfg.r_agent,
                cfg.epsilon,
            )

        trial = current_paths.copy()
        trial[target] = displaced
        feasible, _ = swept_feasible(trial, env, cfg.r_agent, cache)
        phi = compute_phi(trial, env, cfg.r_agent, cfg.dt)
        phi_history.append(phi)

        trajectory = displaced
        if feasible:
            return RepairResult(
                trajectory=trajectory,
                feasible=True,
                passes_used=pass_idx,
                phi_history=phi_history,
            )

    return RepairResult(
        trajectory=trajectory,
        feasible=False,
        passes_used=cfg.K,
        phi_history=phi_history,
    )