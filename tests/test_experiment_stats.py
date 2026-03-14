from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.experiment_stats import (
    aggregate_by_instance,
    build_paper_tables,
    prepare_experiment_dataframe,
    save_experiment_statistics,
    summarize_conditions,
)


def _mock_experiment_runs() -> list[dict[str, object]]:
    return [
        # method A, Narrow, N=4, instance 0
        {
            "method": "CR-PR",
            "env_name": "Narrow",
            "instance_id": 0,
            "n_agents": 4,
            "seed": 0,
            "success": True,
            "t_res_ms": 1200.0,
            "t_e2e_ms": 1800.0,
            "soc_ratio": 1.10,
            "makespan": 4.0,
            "phi": 0.0,
        },
        {
            "method": "CR-PR",
            "env_name": "Narrow",
            "instance_id": 0,
            "n_agents": 4,
            "seed": 1,
            "success": False,
            "t_res_ms": 7000.0,
            "t_e2e_ms": 7300.0,
            "soc_ratio": 1.80,
            "makespan": 8.0,
            "phi": 0.9,
        },
        {
            "method": "CR-PR",
            "env_name": "Narrow",
            "instance_id": 0,
            "n_agents": 4,
            "seed": 2,
            "success": True,
            "t_res_ms": 2000.0,
            "t_e2e_ms": 2600.0,
            "soc_ratio": 1.20,
            "makespan": 5.0,
            "phi": 0.1,
        },
        # method A, Narrow, N=4, instance 1
        {
            "method": "CR-PR",
            "env_name": "Narrow",
            "instance_id": 1,
            "n_agents": 4,
            "seed": 0,
            "success": True,
            "t_res_ms": 3000.0,
            "t_e2e_ms": 3300.0,
            "soc_ratio": 1.05,
            "makespan": 4.5,
            "phi": 0.0,
        },
        {
            "method": "CR-PR",
            "env_name": "Narrow",
            "instance_id": 1,
            "n_agents": 4,
            "seed": 1,
            "success": True,
            "t_res_ms": 3500.0,
            "t_e2e_ms": 3800.0,
            "soc_ratio": 1.08,
            "makespan": 4.8,
            "phi": 0.0,
        },
        # baseline, Narrow, N=4, instance 0
        {
            "method": "LaCAM",
            "env_name": "Narrow",
            "instance_id": 0,
            "n_agents": 4,
            "seed": 0,
            "success": False,
            "t_res_ms": 6000.0,
            "t_e2e_ms": 6100.0,
            "soc_ratio": 1.60,
            "makespan": 7.5,
            "phi": 1.2,
        },
        {
            "method": "LaCAM",
            "env_name": "Narrow",
            "instance_id": 0,
            "n_agents": 4,
            "seed": 1,
            "success": True,
            "t_res_ms": 4900.0,
            "t_e2e_ms": 5001.0,
            "soc_ratio": 1.30,
            "makespan": 6.0,
            "phi": 0.2,
        },
    ]


def test_prepare_experiment_dataframe_adds_capped_runtime_columns() -> None:
    df = prepare_experiment_dataframe(_mock_experiment_runs(), timeout_ms=5000.0)

    assert "t_res_ms_capped" in df.columns
    assert "t_e2e_ms_capped" in df.columns
    assert np.isclose(df.loc[df["seed"] == 1, "t_res_ms_capped"].max(), 5000.0)
    assert (df["t_e2e_ms_capped"] <= 5000.0).any()


def test_prepare_experiment_dataframe_raises_on_missing_required_columns() -> None:
    bad = [{"success": True, "t_res_ms": 1.0}]
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_experiment_dataframe(bad)


def test_prepare_experiment_dataframe_raises_on_negative_runtime() -> None:
    bad = _mock_experiment_runs()
    bad[0] = dict(bad[0])
    bad[0]["t_res_ms"] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        prepare_experiment_dataframe(bad)


def test_aggregate_by_instance_computes_success_rate_and_success_only_means() -> None:
    instance_df = aggregate_by_instance(_mock_experiment_runs(), timeout_ms=5000.0)

    row = instance_df[
        (instance_df["method"] == "CR-PR")
        & (instance_df["env_name"] == "Narrow")
        & (instance_df["n_agents"] == 4)
        & (instance_df["instance_id"] == 0)
    ].iloc[0]

    assert row["n_runs"] == 3
    assert row["n_success"] == 2
    assert np.isclose(row["success_rate"], 2.0 / 3.0)

    # success-only mean: failed run with soc_ratio=1.80 must be excluded
    assert np.isclose(row["mean_soc_ratio_success"], np.mean([1.10, 1.20]))
    assert np.isclose(row["mean_makespan_success"], np.mean([4.0, 5.0]))

    # runtime uses capped value across all runs
    assert np.isclose(row["mean_t_res_ms_capped"], np.mean([1200.0, 5000.0, 2000.0]))


def test_aggregate_by_instance_returns_nan_for_success_only_metrics_when_no_success() -> None:
    runs = [
        {
            "method": "B",
            "env_name": "Office",
            "instance_id": 0,
            "n_agents": 8,
            "seed": 0,
            "success": False,
            "t_res_ms": 7000.0,
            "t_e2e_ms": 7100.0,
            "soc_ratio": 2.0,
            "makespan": 9.0,
        },
        {
            "method": "B",
            "env_name": "Office",
            "instance_id": 0,
            "n_agents": 8,
            "seed": 1,
            "success": False,
            "t_res_ms": 7200.0,
            "t_e2e_ms": 7300.0,
            "soc_ratio": 2.1,
            "makespan": 9.5,
        },
    ]

    instance_df = aggregate_by_instance(runs, timeout_ms=5000.0)
    row = instance_df.iloc[0]

    assert row["n_success"] == 0
    assert np.isnan(row["mean_soc_ratio_success"])
    assert np.isnan(row["mean_makespan_success"])


def test_aggregate_by_instance_raises_on_duplicate_seed_within_group() -> None:
    bad = _mock_experiment_runs() + [
        {
            "method": "CR-PR",
            "env_name": "Narrow",
            "instance_id": 0,
            "n_agents": 4,
            "seed": 0,  # duplicated within same group
            "success": True,
            "t_res_ms": 1500.0,
            "t_e2e_ms": 1700.0,
            "soc_ratio": 1.15,
            "makespan": 4.2,
        }
    ]

    with pytest.raises(ValueError, match="Duplicate seed entries"):
        aggregate_by_instance(bad, timeout_ms=5000.0)


def test_summarize_conditions_aggregates_across_instances() -> None:
    instance_df = aggregate_by_instance(_mock_experiment_runs(), timeout_ms=5000.0)
    condition_df = summarize_conditions(instance_df)

    row = condition_df[
        (condition_df["method"] == "CR-PR")
        & (condition_df["env_name"] == "Narrow")
        & (condition_df["n_agents"] == 4)
    ].iloc[0]

    assert row["n_instances"] == 2

    expected_success_rates = [2.0 / 3.0, 1.0]
    assert np.isclose(row["mean_success_rate"], np.mean(expected_success_rates))
    assert row["std_success_rate"] >= 0.0
    assert row["mean_runtime_res_ms"] >= 0.0
    assert row["mean_soc_ratio_success"] >= 1.0


def test_build_paper_tables_returns_instance_and_condition_tables() -> None:
    instance_df, condition_df = build_paper_tables(_mock_experiment_runs(), timeout_ms=5000.0)

    assert isinstance(instance_df, pd.DataFrame)
    assert isinstance(condition_df, pd.DataFrame)
    assert len(instance_df) == 3  # CR-PR inst0, CR-PR inst1, LaCAM inst0
    assert len(condition_df) == 2  # CR-PR Narrow N4, LaCAM Narrow N4


def test_save_experiment_statistics_writes_all_outputs(tmp_path) -> None:
    runs_csv, instance_csv, condition_csv, summary_json = save_experiment_statistics(
        results=_mock_experiment_runs(),
        output_dir=tmp_path,
        timeout_ms=5000.0,
        stem="paper",
    )

    assert runs_csv.exists()
    assert instance_csv.exists()
    assert condition_csv.exists()
    assert summary_json.exists()

    runs_df = pd.read_csv(runs_csv)
    instance_df = pd.read_csv(instance_csv)
    condition_df = pd.read_csv(condition_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    assert "t_res_ms_capped" in runs_df.columns
    assert "success_rate" in instance_df.columns
    assert "mean_success_rate" in condition_df.columns

    assert summary["timeout_ms"] == 5000.0
    assert summary["n_runs"] == len(_mock_experiment_runs())
    assert summary["n_instance_groups"] == len(instance_df)
    assert summary["n_condition_groups"] == len(condition_df)