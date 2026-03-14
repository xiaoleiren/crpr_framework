from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.environment import Environment
from src.geometry import segment_to_rect_distance


@dataclass
class RRTNode:
    point: np.ndarray
    parent: int
    cost: float


class RRTStar:
    def __init__(
        self,
        env: Environment,
        start: np.ndarray,
        goal: np.ndarray,
        agent_radius: float,
        max_iter: int,
        step_size: float,
        goal_bias: float,
        neighbor_radius: float,
        rng: np.random.Generator,
    ) -> None:
        self.env = env
        self.start = start.astype(float)
        self.goal = goal.astype(float)
        self.agent_radius = agent_radius
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.neighbor_radius = neighbor_radius
        self.rng = rng
        self.nodes: List[RRTNode] = [RRTNode(point=self.start.copy(), parent=-1, cost=0.0)]

    def sample(self) -> np.ndarray:
        if self.rng.random() < self.goal_bias:
            return self.goal.copy()
        return np.array(
            [
                self.rng.uniform(self.agent_radius, self.env.bounds[0] - self.agent_radius),
                self.rng.uniform(self.agent_radius, self.env.bounds[1] - self.agent_radius),
            ],
            dtype=float,
        )

    def nearest_index(self, point: np.ndarray) -> int:
        distances = [np.linalg.norm(node.point - point) for node in self.nodes]
        return int(np.argmin(distances))

    def near_indices(self, point: np.ndarray) -> List[int]:
        return [
            idx
            for idx, node in enumerate(self.nodes)
            if np.linalg.norm(node.point - point) <= self.neighbor_radius
        ]

    def steer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        direction = target - source
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return source.copy()
        step = min(self.step_size, norm)
        return source + direction * (step / norm)

    def segment_is_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        if not self.env.in_bounds(a, self.agent_radius) or not self.env.in_bounds(b, self.agent_radius):
            return False
        for rect in self.env.obstacles:
            if segment_to_rect_distance(a, b, rect) < self.agent_radius - 1e-12:
                return False
        return True

    def reconstruct_path(self, node_index: int) -> np.ndarray:
        path: List[np.ndarray] = []
        idx = node_index
        while idx != -1:
            path.append(self.nodes[idx].point.copy())
            idx = self.nodes[idx].parent
        path.reverse()
        if np.linalg.norm(path[-1] - self.goal) > 1e-6:
            path.append(self.goal.copy())
        return np.vstack(path)

    def direct_fallback(self) -> np.ndarray:
        if self.segment_is_free(self.start, self.goal):
            return np.vstack([self.start, self.goal])

        candidates = [
            np.array([(self.start[0] + self.goal[0]) * 0.5, self.start[1]], dtype=float),
            np.array([(self.start[0] + self.goal[0]) * 0.5, self.goal[1]], dtype=float),
            np.array([self.start[0], self.goal[1]], dtype=float),
            np.array([self.goal[0], self.start[1]], dtype=float),
        ]

        for waypoint in candidates:
            if self.env.collides_with_obstacle(waypoint, self.agent_radius):
                continue
            if self.segment_is_free(self.start, waypoint) and self.segment_is_free(waypoint, self.goal):
                return np.vstack([self.start, waypoint, self.goal])

        raise RuntimeError(
            "RRT* failed to find an obstacle-free fallback path; "
            "increase rrt_max_iter or relax the instance."
        )

    def plan(self) -> np.ndarray:
        if self.segment_is_free(self.start, self.goal):
            return np.vstack([self.start, self.goal])

        for _ in range(self.max_iter):
            x_rand = self.sample()
            nearest_idx = self.nearest_index(x_rand)
            x_new = self.steer(self.nodes[nearest_idx].point, x_rand)
            if not self.segment_is_free(self.nodes[nearest_idx].point, x_new):
                continue

            near = self.near_indices(x_new)
            parent_idx = nearest_idx
            best_cost = self.nodes[nearest_idx].cost + np.linalg.norm(x_new - self.nodes[nearest_idx].point)

            for idx in near:
                candidate_cost = self.nodes[idx].cost + np.linalg.norm(x_new - self.nodes[idx].point)
                if candidate_cost < best_cost and self.segment_is_free(self.nodes[idx].point, x_new):
                    parent_idx = idx
                    best_cost = candidate_cost

            new_index = len(self.nodes)
            self.nodes.append(RRTNode(point=x_new, parent=parent_idx, cost=float(best_cost)))

            for idx in near:
                rewire_cost = best_cost + np.linalg.norm(self.nodes[idx].point - x_new)
                if rewire_cost + 1e-9 < self.nodes[idx].cost and self.segment_is_free(x_new, self.nodes[idx].point):
                    self.nodes[idx].parent = new_index
                    self.nodes[idx].cost = float(rewire_cost)

            if np.linalg.norm(x_new - self.goal) <= self.step_size and self.segment_is_free(x_new, self.goal):
                goal_cost = best_cost + np.linalg.norm(self.goal - x_new)
                self.nodes.append(RRTNode(point=self.goal.copy(), parent=new_index, cost=float(goal_cost)))
                return self.reconstruct_path(len(self.nodes) - 1)

        return self.direct_fallback()