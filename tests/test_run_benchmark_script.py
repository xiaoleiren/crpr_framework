from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.run_benchmark as run_benchmark


def test_run_benchmark_main_writes_expected_outputs(monkeypatch, tmp_path) -> None:
    def fake_run_single(
        env_name: str,
        n_agents: int,
        instance_id: int,
        seed: int,
        cfg_dict: dict,
    ) -> dict:
        return {
            "method": "CR-PR",
            "env_name": env_name,
            "instance_id": instance_id,
            "success": (seed % 2 == 0),
            "t_init_ms": 100.0,
            "t_res_ms": 1200.0 + seed,
            "t_e2e_ms": 1300.0 + seed,
            "phi": 0.0,
            "soc": 10.0,
            "soc_ratio": 1.1,
            "makespan": 4.5,
            "n_agents": n_agents,
            "seed": seed,
            "accepted_candidates": 3,
            "candidate_count": 5,
            "mean_repair_passes": 1.0,
            "outer_iterations": 2,
        }

    monkeypatch.setattr(run_benchmark, "_run_single", fake_run_single)

    output_dir = tmp_path / "benchmark"

    parser = run_benchmark.build_argument_parser()
    args = parser.parse_args(
        [
            "--envs", "Narrow,Office",
            "--n-agents", "2,4",
            "--instances", "2",
            "--seeds", "0,1",
            "--output-dir", str(output_dir),
            "--stem", "crpr",
        ]
    )

    monkeypatch.setattr(run_benchmark, "build_argument_parser", lambda: parser)
    monkeypatch.setattr("sys.argv", ["run_benchmark.py"] + [
        "--envs", "Narrow,Office",
        "--n-agents", "2,4",
        "--instances", "2",
        "--seeds", "0,1",
        "--output-dir", str(output_dir),
        "--stem", "crpr",
    ])

    run_benchmark.main()

    raw_runs_csv = output_dir / "crpr_raw_runs.csv"
    prepared_runs_csv = output_dir / "crpr_runs.csv"
    instance_csv = output_dir / "crpr_instance_summary.csv"
    condition_csv = output_dir / "crpr_condition_summary.csv"
    summary_json = output_dir / "crpr_summary.json"
    manifest_json = output_dir / "crpr_manifest.json"

    assert raw_runs_csv.exists()
    assert prepared_runs_csv.exists()
    assert instance_csv.exists()
    assert condition_csv.exists()
    assert summary_json.exists()
    assert manifest_json.exists()

    raw_df = pd.read_csv(raw_runs_csv)
    prepared_df = pd.read_csv(prepared_runs_csv)
    instance_df = pd.read_csv(instance_csv)
    condition_df = pd.read_csv(condition_csv)
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    # 2 envs * 2 agent settings * 2 instances * 2 seeds
    assert len(raw_df) == 16
    assert len(prepared_df) == 16

    assert {"method", "env_name", "instance_id", "n_agents", "seed"}.issubset(raw_df.columns)
    assert "t_res_ms_capped" in prepared_df.columns
    assert "success_rate" in instance_df.columns
    assert "mean_success_rate" in condition_df.columns

    assert manifest["envs"] == ["Narrow", "Office"]
    assert manifest["n_agents"] == [2, 4]
    assert manifest["instances"] == 2
    assert manifest["seeds"] == [0, 1]