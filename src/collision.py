from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np

from src.environment import Environment
from src.geometry import segment_to_rect_distance, segment_to_segment_distance

Pair = Tuple[int, int]


@dataclass
class CollisionCache:
    active_pairs: Set[Pair]
    refresh_every: int
    iteration: int = 0

    def should_full_scan(self) -> bool:
        return self.iteration % self.refresh_every == 0 or not self.active_pairs

    def step(self) -> None:
        self.iteration += 1


def swept_pair_collision(paths: np.ndarray, i: int, j: int, radius: float) -> bool:
    for k in range(paths.shape[1] - 1):
        if segment_to_segment_distance(
            paths[i, k],
            paths[i, k + 1],
            paths[j, k],
            paths[j, k + 1],
        ) < 2.0 * radius - 1e-12:
            return True
    return False


def swept_obstacle_collision(path: np.ndarray, env: Environment, radius: float) -> bool:
    for k in range(path.shape[0] - 1):
        for rect in env.obstacles:
            if segment_to_rect_distance(path[k], path[k + 1], rect) < radius - 1e-12:
                return True
    return False


def swept_boundary_collision(path: np.ndarray, env: Environment, radius: float) -> bool:
    for k in range(path.shape[0]):
        if not env.in_bounds(path[k], radius):
            return True
    return False


def _segment_cell_keys(a: np.ndarray, b: np.ndarray, cell_size: float) -> Set[Tuple[int, int]]:
    xmin = int(np.floor(min(a[0], b[0]) / cell_size))
    xmax = int(np.floor(max(a[0], b[0]) / cell_size))
    ymin = int(np.floor(min(a[1], b[1]) / cell_size))
    ymax = int(np.floor(max(a[1], b[1]) / cell_size))
    return {(ix, iy) for ix in range(xmin, xmax + 1) for iy in range(ymin, ymax + 1)}


def candidate_pairs_from_broadphase(paths: np.ndarray, radius: float) -> Set[Pair]:
    buckets: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
    cell_size = 2.0 * radius
    for k in range(paths.shape[1] - 1):
        for i in range(paths.shape[0]):
            for cell in _segment_cell_keys(paths[i, k], paths[i, k + 1], cell_size):
                buckets[(k, cell[0], cell[1])].add(i)

    pairs: Set[Pair] = set()
    for agents in buckets.values():
        sorted_agents = sorted(agents)
        for a_idx in range(len(sorted_agents)):
            for b_idx in range(a_idx + 1, len(sorted_agents)):
                pairs.add((sorted_agents[a_idx], sorted_agents[b_idx]))
    return pairs


def swept_feasible(
    paths: np.ndarray,
    env: Environment,
    radius: float,
    cache: Optional[CollisionCache] = None,
) -> Tuple[bool, Set[Pair]]:
    # 1) boundary first: cheapest and globally necessary
    for i in range(paths.shape[0]):
        if swept_boundary_collision(paths[i], env, radius):
            return False, set()

    # 2) agent-agent with broadphase / cache
    if cache is None or cache.should_full_scan():
        pairs = candidate_pairs_from_broadphase(paths, radius)
    else:
        pairs = set(cache.active_pairs)

    colliding_pairs: Set[Pair] = set()
    for i, j in pairs:
        if swept_pair_collision(paths, i, j, radius):
            colliding_pairs.add((i, j))
    if colliding_pairs:
        return False, colliding_pairs

    # 3) agent-obstacle swept check
    for i in range(paths.shape[0]):
        if swept_obstacle_collision(paths[i], env, radius):
            return False, set()

    return True, set()