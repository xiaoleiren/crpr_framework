from __future__ import annotations

from src.config import CRPRConfig
from src.cr_pr import CRPR
from src.environment import Environment


def test_crpr_smoke_run_returns_expected_keys(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)

    result = solver.run()

    expected = {
        "success",
        "t_init_ms",
        "t_res_ms",
        "t_e2e_ms",
        "phi",
        "soc",
        "soc_ratio",
        "makespan",
        "n_agents",
        "seed",
        "accepted_candidates",
        "candidate_count",
        "mean_repair_passes",
        "outer_iterations",
    }
    assert expected.issubset(result.keys())
    assert result["n_agents"] == 2
    assert result["t_e2e_ms"] >= result["t_res_ms"]


def test_minimum_waypoint_clearance_counts_boundary_as_well(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)

    path = solver.paths[0].copy()
    path[1] = [cfg.r_agent + 0.02, 2.5]

    clearance = solver._minimum_waypoint_clearance(path)

    assert clearance <= 0.02 + 1e-9


def test_try_candidate_no_repair_branch_uses_cache_compatible_signature(small_cfg_dict: dict) -> None:
    cfg_data = dict(small_cfg_dict)
    cfg_data["ablation"] = "no_repair"
    cfg = CRPRConfig.from_mapping(cfg_data)

    env = Environment.build("Narrow")
    solver = CRPR(env=env, n_agents=2, cfg=cfg, seed=0)

    target = 0
    candidate = solver.paths[target].copy()

    ok, repair = solver._try_candidate(target, candidate)

    assert isinstance(ok, bool)
    if ok:
        assert repair is not None
        assert repair.trajectory is not None