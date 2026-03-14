from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.geometry import Rectangle, point_to_rect_distance

logger = logging.getLogger(__name__)


@dataclass
class Environment:
    name: str
    width: float
    height: float
    obstacles: list[Rectangle]

    @property
    def bounds(self) -> tuple[float, float]:
        """
        Backward-compatible view used by existing tests and code.
        """
        return (self.width, self.height)

    @classmethod
    def build(cls, name: str) -> "Environment":
        if name == "Narrow":
            bounds = (20.0, 5.0)
            obstacles = [
                Rectangle(5.0, 0.0, 6.0, 1.9),
                Rectangle(5.0, 3.1, 6.0, 5.0),
                Rectangle(14.0, 0.0, 15.0, 1.9),
                Rectangle(14.0, 3.1, 15.0, 5.0),
            ]
        elif name == "Office":
            bounds = (50.0, 50.0)
            rects: List[Rectangle] = []
            rng = np.random.default_rng(123)
            for cx in np.linspace(6.0, 44.0, 5):
                for cy in np.linspace(6.0, 44.0, 8):
                    radius = float(rng.uniform(0.3, 0.8))
                    rects.append(Rectangle(cx - radius, cy - radius, cx + radius, cy + radius))
            obstacles = rects[:40]
        elif name in {"C.W.", "CW", "Warehouse"}:
            canonical_name = "Warehouse"
            bounds = (30.0, 20.0)
            obstacles = [
                Rectangle(0.0, 0.0, 30.0, 1.0),
                Rectangle(0.0, 19.0, 30.0, 20.0),
                Rectangle(4.5, 4.0, 5.5, 16.0),
                Rectangle(8.5, 4.0, 9.5, 16.0),
                Rectangle(12.5, 4.0, 13.5, 16.0),
                Rectangle(16.5, 4.0, 17.5, 16.0),
                Rectangle(20.5, 4.0, 21.5, 16.0),
                Rectangle(24.5, 4.0, 25.5, 16.0),
            ]
            name = canonical_name
        else:
            raise ValueError(f"Unknown environment: {name}")

        env = cls(
            name=name,
            width=float(bounds[0]),
            height=float(bounds[1]),
            obstacles=list(obstacles),
        )
        logger.info("Loaded environment %s with %d obstacles.", env.name, len(env.obstacles))
        return env

    def in_bounds(self, point: np.ndarray, margin: float = 0.0) -> bool:
        return bool(
            margin <= point[0] <= self.width - margin
            and margin <= point[1] <= self.height - margin
        )

    def project_to_bounds(self, point: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.array(
            [
                np.clip(point[0], margin, self.width - margin),
                np.clip(point[1], margin, self.height - margin),
            ],
            dtype=float,
        )

    def boundary_clearance(self, point: np.ndarray) -> float:
        return float(
            min(
                point[0],
                self.width - point[0],
                point[1],
                self.height - point[1],
            )
        )

    def collides_with_obstacle(self, point: np.ndarray, radius: float = 0.0) -> bool:
        for rect in self.obstacles:
            if radius > 0.0:
                if point_to_rect_distance(point, rect) < radius - 1e-12:
                    return True
            else:
                if rect.contains(point):
                    return True
        return False

    def sample_free_points(
        self,
        n: int,
        radius: float,
        rng: np.random.Generator,
        min_pair_distance: float,
        max_tries: int = 50_000,
    ) -> np.ndarray:
        points: List[np.ndarray] = []
        tries = 0

        while len(points) < n and tries < max_tries:
            tries += 1
            point = np.array(
                [
                    rng.uniform(radius, self.width - radius),
                    rng.uniform(radius, self.height - radius),
                ],
                dtype=float,
            )

            if self.collides_with_obstacle(point, radius):
                continue

            if any(np.linalg.norm(point - other) < min_pair_distance for other in points):
                continue

            points.append(point)

        if len(points) != n:
            raise RuntimeError("Could not sample enough free points under the required separation.")

        return np.vstack(points)

    def sample_start_goal(
        self,
        n: int,
        radius: float,
        seed: int,
        min_goal_distance: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        starts = self.sample_free_points(
            n=n,
            radius=radius,
            rng=rng,
            min_pair_distance=2.2 * radius,
        )
        goals = self.sample_free_points(
            n=n,
            radius=radius,
            rng=rng,
            min_pair_distance=2.2 * radius,
        )

        for i in range(n):
            attempt = 0
            while np.linalg.norm(starts[i] - goals[i]) < min_goal_distance and attempt < 1_000:
                attempt += 1
                goals[i] = self.sample_free_points(
                    n=1,
                    radius=radius,
                    rng=rng,
                    min_pair_distance=0.0,
                )[0]

            if np.linalg.norm(starts[i] - goals[i]) < min_goal_distance:
                raise RuntimeError("Could not sample sufficiently separated start/goal pair.")

        return starts, goals