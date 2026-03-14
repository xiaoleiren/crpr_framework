from __future__ import annotations

from src.config import CRPRConfig
from src.cr_pr import CRPR
from src.environment import Environment


def test_crpr_run_includes_experiment_metadata(small_cfg_dict: dict) -> None:
    cfg = CRPRConfig.from_mapping(small_cfg_dict)
    env = Environment.build("Narrow")

    solver = CRPR(
        env=env,
        n_agents=2,
        cfg=cfg,
        seed=7,
        instance_id=3,
    )
    result = solver.run()

    assert "method" in result
    assert "env_name" in result
    assert "instance_id" in result

    assert result["method"] == "CR-PR"
    assert result["env_name"] == "Narrow"
    assert result["instance_id"] == 3
    assert result["seed"] == 7
    assert result["n_agents"] == 2