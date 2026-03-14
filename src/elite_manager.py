from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.environment import Environment
from src.geometry import pairwise_sup_distance
from src.objective import LexicographicScore, lexicographically_better, local_score


@dataclass
class EliteMember:
    trajectory: np.ndarray
    score: LexicographicScore


class EliteManager:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.members: List[EliteMember] = []

    def __len__(self) -> int:
        return len(self.members)

    def mean_diversity_without(self, member_index: int) -> float:
        if len(self.members) <= 1:
            return 0.0
        member = self.members[member_index].trajectory
        distances = [
            pairwise_sup_distance(member, other.trajectory)
            for idx, other in enumerate(self.members)
            if idx != member_index
        ]
        return float(np.mean(distances)) if distances else 0.0

    def mean_diversity_to_others(self, trajectory: np.ndarray, exclude_index: int | None = None) -> float:
        distances = []
        for idx, other in enumerate(self.members):
            if exclude_index is not None and idx == exclude_index:
                continue
            distances.append(pairwise_sup_distance(trajectory, other.trajectory))
        return float(np.mean(distances)) if distances else 0.0

    def insert(
        self,
        candidate: np.ndarray,
        target: int,
        paths: np.ndarray,
        env: Environment,
        radius: float,
        dt: float,
    ) -> None:
        candidate_score = local_score(candidate, paths, target, env, radius, dt)
        member = EliteMember(trajectory=candidate.copy(), score=candidate_score)
        if len(self.members) < self.max_size:
            self.members.append(member)
            return

        redundant_idx = int(np.argmin([self.mean_diversity_without(i) for i in range(len(self.members))]))
        redundant_member = self.members[redundant_idx]
        candidate_diversity = self.mean_diversity_to_others(candidate, exclude_index=redundant_idx)
        redundant_diversity = self.mean_diversity_without(redundant_idx)
        if candidate_diversity > redundant_diversity and (
            lexicographically_better(candidate_score, redundant_member.score)
            or candidate_score.as_tuple() == redundant_member.score.as_tuple()
        ):
            self.members[redundant_idx] = member

    def diverse_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.members:
            raise RuntimeError("EliteManager is empty.")
        if len(self.members) == 1:
            only = self.members[0].trajectory.copy()
            return only, only.copy()
        best_pair = (0, 1)
        best_distance = -np.inf
        for i in range(len(self.members)):
            for j in range(i + 1, len(self.members)):
                distance = pairwise_sup_distance(self.members[i].trajectory, self.members[j].trajectory)
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (i, j)
        return (
            self.members[best_pair[0]].trajectory.copy(),
            self.members[best_pair[1]].trajectory.copy(),
        )
