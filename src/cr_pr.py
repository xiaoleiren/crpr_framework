from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.collision import CollisionCache, swept_feasible
from src.config import CRPRConfig
from src.elite_manager import EliteManager
from src.environment import Environment
from src.geometry import compute_makespan, pairwise_sup_distance, time_align_paths
from src.kinematic import bidirectional_kinematic_projection
from src.objective import (
    compute_phi,
    global_score,
    lexicographically_better,
    select_target_agent,
    sum_of_costs,
)
from src.repair_operator import RepairResult, repair_l_beta
from src.rrt_star import RRTStar

logger = logging.getLogger(__name__)


@dataclass
class PropositionConditions:
    clearance_satisfied: bool
    step_bound_satisfied: bool
    epsilon_net: float
    diversity_diameter: float
    sampled_relink_clearance: float


class CRPR:
    def __init__(
        self,
        env: Environment,
        n_agents: int,
        cfg: CRPRConfig,
        seed: int = 0,
        instance_id: int | None = None,
    ) -> None:
        self.env = env
        self.n_agents = n_agents
        self.cfg = cfg
        self.seed = seed
        self.instance_id = instance_id

        # Best-effort environment naming for experiment logging.
        self.env_name = getattr(env, "name", None)
        if self.env_name is None:
            self.env_name = env.__class__.__name__

        self.rng = np.random.default_rng(seed)
        self.starts, self.goals = env.sample_start_goal(n_agents, cfg.r_agent, seed)

        init_start = time.perf_counter()
        polylines = self._plan_independent_paths()
        self.paths, self.arrival_times, self.t_max = time_align_paths(polylines, cfg.dt, cfg.v_max)
        self.initial_paths = self.paths.copy()
        self.initial_soc_lb = sum_of_costs(self.initial_paths)
        self.initialization_ms = (time.perf_counter() - init_start) * 1000.0

        self.elites = [EliteManager(cfg.elite_max) for _ in range(n_agents)]
        self._initialize_elites()
        self.collision_cache = CollisionCache(active_pairs=set(), refresh_every=cfg.cache_refresh_every)

    def _plan_independent_paths(self) -> List[np.ndarray]:
        polylines: List[np.ndarray] = []
        for i in range(self.n_agents):
            planner = RRTStar(
                env=self.env,
                start=self.starts[i],
                goal=self.goals[i],
                agent_radius=self.cfg.r_agent,
                max_iter=self.cfg.rrt_max_iter,
                step_size=self.cfg.rrt_step_size,
                goal_bias=self.cfg.rrt_goal_bias,
                neighbor_radius=self.cfg.rrt_neighbor_radius,
                rng=self.rng,
            )
            polylines.append(planner.plan())
        return polylines

    def _initialize_elites(self) -> None:
        for i in range(self.n_agents):
            self.elites[i].insert(self.paths[i], i, self.paths, self.env, self.cfg.r_agent, self.cfg.dt)
            for _ in range(self.cfg.elite_max - 1):
                noisy = self.paths[i].copy()
                if noisy.shape[0] > 2:
                    noise = self.rng.normal(0.0, self.cfg.init_noise_sigma, size=noisy[1:-1].shape)
                    noisy[1:-1] += noise
                projected = bidirectional_kinematic_projection(noisy, self.starts[i], self.goals[i], self.cfg)
                self.elites[i].insert(projected, i, self.paths, self.env, self.cfg.r_agent, self.cfg.dt)

    def _minimum_waypoint_clearance(self, path: np.ndarray) -> float:
        clearance = np.inf
        for point in path:
            if self.env.obstacles:
                obs_clearance = min(
                    np.linalg.norm(point - rect.nearest_point(point)) - self.cfg.r_agent
                    for rect in self.env.obstacles
                )
            else:
                obs_clearance = np.inf

            boundary_clearance = self.env.boundary_clearance(point) - self.cfg.r_agent
            clearance = min(clearance, obs_clearance, boundary_clearance)

        return float(clearance)

    def _estimate_relink_segment_clearance(
        self,
        init_path: np.ndarray,
        guide_path: np.ndarray,
    ) -> float:
        """
        Paper-aligned heuristic:
        sample the relinking segment over alpha = 0, Δα, ..., 1,
        then measure the minimum waypoint/body-clearance over that sampled set.
        This is a cheap heuristic, not the exact theorem-side δ_clear test.
        """
        min_clearance = np.inf
        for alpha in self._alpha_schedule():
            candidate = (1.0 - alpha) * init_path + alpha * guide_path
            min_clearance = min(min_clearance, self._minimum_waypoint_clearance(candidate))
        return float(min_clearance)

    def proposition_condition_summary(self, init_path: np.ndarray, guide_path: np.ndarray) -> PropositionConditions:
        diversity_diameter = pairwise_sup_distance(init_path, guide_path)
        epsilon_net = self.cfg.delta_alpha * diversity_diameter

        sampled_relink_clearance = self._estimate_relink_segment_clearance(init_path, guide_path)
        clearance_satisfied = sampled_relink_clearance > max(2.0 * epsilon_net, self.cfg.min_clearance_skip)

        step_bound = 0.0
        if diversity_diameter > 1e-12:
            local_lipschitz_proxy = 1.0 / max(sampled_relink_clearance, self.cfg.epsilon)
            step_bound = sampled_relink_clearance / (
                2.0 * (1.0 + self.cfg.beta * local_lipschitz_proxy) * diversity_diameter
            )

        step_bound_satisfied = self.cfg.delta_alpha <= step_bound if step_bound > 0.0 else False
        return PropositionConditions(
            clearance_satisfied=clearance_satisfied,
            step_bound_satisfied=step_bound_satisfied,
            epsilon_net=epsilon_net,
            diversity_diameter=diversity_diameter,
            sampled_relink_clearance=sampled_relink_clearance,
        )

    def _alpha_schedule(self) -> np.ndarray:
        if self.cfg.ablation == "no_relink":
            return np.array([0.0, 1.0], dtype=float)
        n_steps = int(round(1.0 / self.cfg.delta_alpha))
        return np.linspace(0.0, 1.0, n_steps + 1)

    def _try_candidate(self, target: int, candidate: np.ndarray) -> Tuple[bool, RepairResult | None]:
        if self.cfg.ablation == "no_repair":
            trial = self.paths.copy()
            trial[target] = candidate
            feasible, colliding_pairs = swept_feasible(
                trial,
                self.env,
                self.cfg.r_agent,
                self.collision_cache,
            )
            self.collision_cache.active_pairs = colliding_pairs
            if feasible:
                return True, RepairResult(
                    trajectory=candidate,
                    feasible=True,
                    passes_used=0,
                    phi_history=[],
                )
            return False, None

        repair = repair_l_beta(
            candidate,
            self.paths,
            target,
            self.goals,
            self.env,
            self.cfg,
            cache=self.collision_cache,
        )
        if repair.trajectory is None:
            return False, repair
        return True, repair

    def run(self) -> Dict[str, float | int | bool | str | None]:
        backend_start = time.perf_counter()
        accepted_candidates = 0
        accepted_passes: List[int] = []
        candidate_count = 0
        outer_iterations = 0

        current_score = global_score(self.paths, self.env, self.cfg.r_agent, self.cfg.dt)
        feasible, colliding_pairs = swept_feasible(
            self.paths,
            self.env,
            self.cfg.r_agent,
            self.collision_cache,
        )
        self.collision_cache.active_pairs = colliding_pairs

        while not feasible and (time.perf_counter() - backend_start) < self.cfg.timeout_s:
            outer_iterations += 1
            self.collision_cache.step()

            target = select_target_agent(self.paths, self.env, self.cfg.r_agent, self.cfg.dt)
            init_path, guide_path = self.elites[target].diverse_pair()
            conditions = self.proposition_condition_summary(init_path, guide_path)

            # Paper Algorithm 1: skip elite pair if sampled relinking segment is too close to obstacles/walls.
            if conditions.sampled_relink_clearance < self.cfg.min_clearance_skip:
                logger.debug(
                    "Skipping elite pair for agent %d because sampled relinking clearance %.6f < %.6f.",
                    target,
                    conditions.sampled_relink_clearance,
                    self.cfg.min_clearance_skip,
                )
                continue

            improved_this_round = False
            for alpha in self._alpha_schedule():
                candidate_count += 1
                candidate = (1.0 - alpha) * init_path + alpha * guide_path

                ok, repair = self._try_candidate(target, candidate)
                if not ok or repair is None or repair.trajectory is None:
                    continue

                new_path = repair.trajectory
                trial = self.paths.copy()
                trial[target] = new_path
                trial_score = global_score(trial, self.env, self.cfg.r_agent, self.cfg.dt)

                if lexicographically_better(trial_score, current_score):
                    self.paths = trial
                    current_score = trial_score
                    accepted_candidates += 1
                    accepted_passes.append(repair.passes_used)
                    improved_this_round = True

                    feasible, colliding_pairs = swept_feasible(
                        self.paths,
                        self.env,
                        self.cfg.r_agent,
                        self.collision_cache,
                    )
                    self.collision_cache.active_pairs = colliding_pairs

                    self.elites[target].insert(
                        new_path,
                        target,
                        self.paths,
                        self.env,
                        self.cfg.r_agent,
                        self.cfg.dt,
                    )

                    if feasible:
                        break
                else:
                    self.elites[target].insert(
                        new_path,
                        target,
                        self.paths,
                        self.env,
                        self.cfg.r_agent,
                        self.cfg.dt,
                    )

            if not improved_this_round:
                feasible, colliding_pairs = swept_feasible(
                    self.paths,
                    self.env,
                    self.cfg.r_agent,
                    self.collision_cache,
                )
                self.collision_cache.active_pairs = colliding_pairs

        backend_ms = (time.perf_counter() - backend_start) * 1000.0
        total_soc = sum_of_costs(self.paths)
        makespan = compute_makespan(self.paths, self.goals, self.cfg.dt)

        return {
            "method": "CR-PR",
            "env_name": self.env_name,
            "instance_id": self.instance_id,
            "success": feasible,
            "t_init_ms": float(self.initialization_ms),
            "t_res_ms": float(backend_ms),
            "t_e2e_ms": float(self.initialization_ms + backend_ms),
            "phi": float(compute_phi(self.paths, self.env, self.cfg.r_agent, self.cfg.dt)),
            "soc": float(total_soc),
            "soc_ratio": float(total_soc / max(self.initial_soc_lb, 1e-9)),
            "makespan": float(makespan),
            "n_agents": int(self.n_agents),
            "seed": int(self.seed),
            "accepted_candidates": int(accepted_candidates),
            "candidate_count": int(candidate_count),
            "mean_repair_passes": float(np.mean(accepted_passes)) if accepted_passes else 0.0,
            "outer_iterations": int(outer_iterations),
        }