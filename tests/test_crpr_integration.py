from __future__ import annotations

import numpy as np

from src.config import CRPRConfig
from src.cr_pr import CRPR
from src.environment import Environment
from src.objective import compute_phi, global_score, select_target_agent


def test_crpr_initial_score_matches_objective_stack(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)

    score = global_score(solver.paths, env, cfg.r_agent, cfg.dt)

    assert np.isclose(score.phi, compute_phi(solver.paths, env, cfg.r_agent, cfg.dt))
    assert score.soc >= 0.0


def test_crpr_select_target_agent_returns_valid_index_on_initialized_paths(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=3, cfg=cfg, seed=0)

    target = select_target_agent(solver.paths, env, cfg.r_agent, cfg.dt)
    assert 0 <= target < 3


def test_crpr_run_does_not_increase_phi_relative_to_initial_state(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)

    initial_phi = compute_phi(solver.paths, env, cfg.r_agent, cfg.dt)
    result = solver.run()
    final_phi = compute_phi(solver.paths, env, cfg.r_agent, cfg.dt)

    assert np.isclose(final_phi, result["phi"])
    assert final_phi <= initial_phi + 1e-9


def test_crpr_run_keeps_paths_kinematically_and_boundary_consistent(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)

    solver.run()

    for i in range(solver.paths.shape[0]):
        assert np.allclose(solver.paths[i, 0], solver.starts[i])
        assert np.allclose(solver.paths[i, -1], solver.goals[i])
        for point in solver.paths[i]:
            assert env.in_bounds(point, cfg.r_agent)


def test_crpr_cache_object_is_live_after_run(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)

    initial_iteration = solver.collision_cache.iteration
    solver.run()

    assert solver.collision_cache.iteration >= initial_iteration
    assert isinstance(solver.collision_cache.active_pairs, set)


def test_crpr_no_repair_ablation_still_returns_well_formed_metrics(small_cfg_dict: dict) -> None:
    cfg_map = dict(small_cfg_dict)
    cfg_map["ablation"] = "no_repair"
    cfg = CRPRConfig.from_mapping(cfg_map)

    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)
    result = solver.run()

    assert "phi" in result
    assert "soc" in result
    assert "candidate_count" in result
    assert result["candidate_count"] >= 0