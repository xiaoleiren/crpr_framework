from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    bca_confidence_interval,
    mcnemar_test,
    paired_runtime_reduction,
    save_run_statistics,
    summarize_results,
    wilcoxon_runtime_test,
)


def _mock_results() -> list[dict[str, float | int | bool]]:
    return [
        {
            "success": True,
            "t_init_ms": 10.0,
            "t_res_ms": 20.0,
            "t_e2e_ms": 30.0,
            "phi": 0.0,
            "soc": 12.0,
            "soc_ratio": 1.05,
            "makespan": 5.0,
            "n_agents": 4,
            "seed": 0,
            "accepted_candidates": 3,
            "candidate_count": 10,
            "mean_repair_passes": 1.5,
            "outer_iterations": 2,
        },
        {
            "success": False,
            "t_init_ms": 11.0,
            "t_res_ms": 40.0,
            "t_e2e_ms": 51.0,
            "phi": 0.8,
            "soc": 13.0,
            "soc_ratio": 1.15,
            "makespan": 5.5,
            "n_agents": 4,
            "seed": 1,
            "accepted_candidates": 1,
            "candidate_count": 8,
            "mean_repair_passes": 2.0,
            "outer_iterations": 3,
        },
        {
            "success": True,
            "t_init_ms": 9.0,
            "t_res_ms": 25.0,
            "t_e2e_ms": 34.0,
            "phi": 0.1,
            "soc": 11.0,
            "soc_ratio": 1.00,
            "makespan": 4.8,
            "n_agents": 4,
            "seed": 2,
            "accepted_candidates": 4,
            "candidate_count": 11,
            "mean_repair_passes": 1.0,
            "outer_iterations": 2,
        },
    ]


def test_summarize_results_returns_dataframe_with_expected_columns() -> None:
    df = summarize_results(_mock_results())

    assert isinstance(df, pd.DataFrame)
    expected_columns = {
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
    assert expected_columns.issubset(df.columns)
    assert len(df) == 3


def test_mcnemar_test_counts_discordant_pairs_correctly() -> None:
    success_a = [True, True, False, False, True]
    success_b = [True, False, True, False, False]

    result = mcnemar_test(success_a, success_b)

    assert result["discordant_a_only"] == 2
    assert result["discordant_b_only"] == 1
    assert result["n_discordant"] == 3
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["odds_ratio"] == 2.0


def test_mcnemar_test_returns_trivial_result_when_no_discordant_pairs() -> None:
    success_a = [True, False, True]
    success_b = [True, False, True]

    result = mcnemar_test(success_a, success_b)

    assert result["discordant_a_only"] == 0
    assert result["discordant_b_only"] == 0
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0
    assert result["odds_ratio"] is None


def test_paired_runtime_reduction_matches_expected_formula() -> None:
    crpr = [8.0, 10.0, 12.0]
    baseline = [10.0, 10.0, 20.0]

    reductions = paired_runtime_reduction(crpr, baseline)

    assert np.allclose(reductions, [0.2, 0.0, 0.4])


def test_paired_runtime_reduction_handles_zero_baseline_with_guard() -> None:
    crpr = [1.0, 2.0]
    baseline = [0.0, 4.0]

    reductions = paired_runtime_reduction(crpr, baseline)

    assert reductions.shape == (2,)
    assert np.isfinite(reductions).all()


def test_wilcoxon_runtime_test_returns_neutral_result_for_all_zero_reductions() -> None:
    reductions = [0.0, 0.0, 0.0, 0.0]

    result = wilcoxon_runtime_test(reductions)

    assert result["n"] == 0
    assert result["p_value"] == 1.0
    assert result["effect_r"] == 0.0


def test_wilcoxon_runtime_test_returns_valid_statistics_for_nonzero_reductions() -> None:
    reductions = [0.2, 0.1, -0.05, 0.3, 0.15]

    result = wilcoxon_runtime_test(reductions)

    assert result["n"] == 5
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["effect_r"] >= 0.0


def test_bca_confidence_interval_returns_mean_and_ordered_interval() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]

    result = bca_confidence_interval(samples, confidence_level=0.90, n_resamples=2000)

    assert np.isclose(result["mean"], 3.0)
    assert result["low"] <= result["mean"] <= result["high"]


def test_bca_confidence_interval_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bca_confidence_interval([])

def test_mcnemar_test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        mcnemar_test([True, False], [True])


def test_paired_runtime_reduction_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_runtime_reduction([1.0, 2.0], [1.0])

def test_save_run_statistics_writes_csv_and_json_summary(tmp_path) -> None:
    results = _mock_results()

    csv_path, json_path = save_run_statistics(
        results=results,
        output_dir=tmp_path,
        env_name="Narrow",
        n_agents=4,
    )

    assert csv_path.exists()
    assert json_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert "success" in df.columns
    assert "soc_ratio" in df.columns

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    expected_summary_keys = {
        "success_rate",
        "mean_t_res_ms",
        "mean_t_e2e_ms",
        "mean_soc_ratio",
        "mean_makespan",
    }
    assert expected_summary_keys == set(summary.keys())

    assert np.isclose(summary["success_rate"], 2.0 / 3.0)
    assert np.isclose(summary["mean_t_res_ms"], np.mean([20.0, 40.0, 25.0]))
    assert np.isclose(summary["mean_t_e2e_ms"], np.mean([30.0, 51.0, 34.0]))
    assert np.isclose(summary["mean_soc_ratio"], np.mean([1.05, 1.15, 1.00]))
    assert np.isclose(summary["mean_makespan"], np.mean([5.0, 5.5, 4.8]))


def test_save_run_statistics_creates_output_directory_if_missing(tmp_path) -> None:
    nested = tmp_path / "reports" / "exp1"

    csv_path, json_path = save_run_statistics(
        results=_mock_results(),
        output_dir=nested,
        env_name="Office",
        n_agents=8,
    )

    assert nested.exists()
    assert csv_path.parent == nested
    assert json_path.parent == nested
    assert csv_path.name == "Office_N8.csv"
    assert json_path.name == "Office_N8_summary.json"