from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_RUN_COLUMNS = {
    "success",
    "t_res_ms",
    "t_e2e_ms",
    "soc_ratio",
    "makespan",
    "n_agents",
    "seed",
}

DEFAULT_INSTANCE_GROUP_COLUMNS = ("method", "env_name", "n_agents", "instance_id")
DEFAULT_CONDITION_GROUP_COLUMNS = ("method", "env_name", "n_agents")


def _as_dataframe(results: Sequence[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        df = results.copy()
    else:
        df = pd.DataFrame(results)

    if df.empty:
        raise ValueError("Results are empty.")

    return df


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _runtime_cap(series: pd.Series, timeout_ms: float) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").astype(float)
    return values.clip(upper=float(timeout_ms))


def prepare_experiment_dataframe(
    results: Sequence[Mapping[str, object]] | pd.DataFrame,
    timeout_ms: float = 5000.0,
) -> pd.DataFrame:
    """
    Normalize run-level experiment results and add capped runtime columns.

    Required columns:
        success, t_res_ms, t_e2e_ms, soc_ratio, makespan, n_agents, seed

    Strongly recommended metadata columns:
        method, env_name, instance_id
    """
    if timeout_ms <= 0.0:
        raise ValueError("timeout_ms must be positive.")

    df = _as_dataframe(results)
    _require_columns(df, REQUIRED_RUN_COLUMNS)

    out = df.copy()

    out["success"] = out["success"].astype(bool)
    out["t_res_ms"] = pd.to_numeric(out["t_res_ms"], errors="raise").astype(float)
    out["t_e2e_ms"] = pd.to_numeric(out["t_e2e_ms"], errors="raise").astype(float)
    out["soc_ratio"] = pd.to_numeric(out["soc_ratio"], errors="raise").astype(float)
    out["makespan"] = pd.to_numeric(out["makespan"], errors="raise").astype(float)
    out["n_agents"] = pd.to_numeric(out["n_agents"], errors="raise").astype(int)
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)

    if "phi" in out.columns:
        out["phi"] = pd.to_numeric(out["phi"], errors="raise").astype(float)

    if "method" in out.columns:
        out["method"] = out["method"].astype(str)
    if "env_name" in out.columns:
        out["env_name"] = out["env_name"].astype(str)
    if "instance_id" in out.columns:
        out["instance_id"] = pd.to_numeric(out["instance_id"], errors="raise").astype("Int64")

    if (out["t_res_ms"] < 0.0).any() or (out["t_e2e_ms"] < 0.0).any():
        raise ValueError("Runtime columns must be non-negative.")

    out["t_res_ms_capped"] = _runtime_cap(out["t_res_ms"], timeout_ms)
    out["t_e2e_ms_capped"] = _runtime_cap(out["t_e2e_ms"], timeout_ms)
    out["timeout_ms"] = float(timeout_ms)

    return out


def aggregate_by_instance(
    results: Sequence[Mapping[str, object]] | pd.DataFrame,
    timeout_ms: float = 5000.0,
    group_cols: Sequence[str] = DEFAULT_INSTANCE_GROUP_COLUMNS,
) -> pd.DataFrame:
    """
    Aggregate multiple seed runs for each instance.

    Expected grouping keys by default:
        method, env_name, n_agents, instance_id

    Output metrics:
        - n_runs
        - n_success
        - success_rate
        - mean/std capped runtimes across all runs
        - mean/std soc_ratio on successful runs only
        - mean/std makespan on successful runs only
        - mean/std phi on successful runs only (if present)
    """
    df = prepare_experiment_dataframe(results, timeout_ms=timeout_ms)
    _require_columns(df, group_cols)

    duplicate_mask = df.duplicated(subset=[*group_cols, "seed"], keep=False)
    if duplicate_mask.any():
        duplicated_rows = df.loc[duplicate_mask, [*group_cols, "seed"]]
        raise ValueError(
            "Duplicate seed entries found within an instance group: "
            f"{duplicated_rows.to_dict(orient='records')}"
        )

    def _agg(group: pd.DataFrame) -> pd.Series:
        success_mask = group["success"].to_numpy(dtype=bool)
        success_df = group.loc[success_mask]

        row: dict[str, object] = {
            "n_runs": int(len(group)),
            "n_success": int(success_mask.sum()),
            "success_rate": float(success_mask.mean()),
            "mean_t_res_ms_capped": float(group["t_res_ms_capped"].mean()),
            "std_t_res_ms_capped": float(group["t_res_ms_capped"].std(ddof=0)),
            "mean_t_e2e_ms_capped": float(group["t_e2e_ms_capped"].mean()),
            "std_t_e2e_ms_capped": float(group["t_e2e_ms_capped"].std(ddof=0)),
            "timeout_ms": float(group["timeout_ms"].iloc[0]),
        }

        if len(success_df) == 0:
            row["mean_soc_ratio_success"] = np.nan
            row["std_soc_ratio_success"] = np.nan
            row["mean_makespan_success"] = np.nan
            row["std_makespan_success"] = np.nan
            if "phi" in group.columns:
                row["mean_phi_success"] = np.nan
                row["std_phi_success"] = np.nan
        else:
            row["mean_soc_ratio_success"] = float(success_df["soc_ratio"].mean())
            row["std_soc_ratio_success"] = float(success_df["soc_ratio"].std(ddof=0))
            row["mean_makespan_success"] = float(success_df["makespan"].mean())
            row["std_makespan_success"] = float(success_df["makespan"].std(ddof=0))

            if "phi" in group.columns:
                row["mean_phi_success"] = float(success_df["phi"].mean())
                row["std_phi_success"] = float(success_df["phi"].std(ddof=0))

        return pd.Series(row)

    aggregated = (
        df.groupby(list(group_cols), dropna=False, sort=True)
        .apply(_agg)
        .reset_index()
    )

    return aggregated


def summarize_conditions(
    instance_summary: pd.DataFrame,
    group_cols: Sequence[str] = DEFAULT_CONDITION_GROUP_COLUMNS,
) -> pd.DataFrame:
    """
    Aggregate instance-level summaries into condition-level table.

    Typical condition grouping:
        method, env_name, n_agents
    """
    df = instance_summary.copy()
    _require_columns(
        df,
        [
            *group_cols,
            "success_rate",
            "mean_t_res_ms_capped",
            "mean_t_e2e_ms_capped",
            "mean_soc_ratio_success",
            "mean_makespan_success",
        ],
    )

    def _summary(group: pd.DataFrame) -> pd.Series:
        out: dict[str, object] = {
            "n_instances": int(len(group)),
            "mean_success_rate": float(group["success_rate"].mean()),
            "std_success_rate": float(group["success_rate"].std(ddof=0)),
            "mean_runtime_res_ms": float(group["mean_t_res_ms_capped"].mean()),
            "std_runtime_res_ms": float(group["mean_t_res_ms_capped"].std(ddof=0)),
            "mean_runtime_e2e_ms": float(group["mean_t_e2e_ms_capped"].mean()),
            "std_runtime_e2e_ms": float(group["mean_t_e2e_ms_capped"].std(ddof=0)),
            "mean_soc_ratio_success": float(group["mean_soc_ratio_success"].mean(skipna=True)),
            "std_soc_ratio_success": float(group["mean_soc_ratio_success"].std(ddof=0, skipna=True)),
            "mean_makespan_success": float(group["mean_makespan_success"].mean(skipna=True)),
            "std_makespan_success": float(group["mean_makespan_success"].std(ddof=0, skipna=True)),
        }

        if "mean_phi_success" in group.columns:
            out["mean_phi_success"] = float(group["mean_phi_success"].mean(skipna=True))
            out["std_phi_success"] = float(group["mean_phi_success"].std(ddof=0, skipna=True))

        if "timeout_ms" in group.columns:
            out["timeout_ms"] = float(group["timeout_ms"].iloc[0])

        return pd.Series(out)

    return (
        df.groupby(list(group_cols), dropna=False, sort=True)
        .apply(_summary)
        .reset_index()
    )


def build_paper_tables(
    results: Sequence[Mapping[str, object]] | pd.DataFrame,
    timeout_ms: float = 5000.0,
    instance_group_cols: Sequence[str] = DEFAULT_INSTANCE_GROUP_COLUMNS,
    condition_group_cols: Sequence[str] = DEFAULT_CONDITION_GROUP_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
        run-level results -> instance summary -> condition summary
    """
    instance_df = aggregate_by_instance(
        results=results,
        timeout_ms=timeout_ms,
        group_cols=instance_group_cols,
    )
    condition_df = summarize_conditions(
        instance_summary=instance_df,
        group_cols=condition_group_cols,
    )
    return instance_df, condition_df


def save_experiment_statistics(
    results: Sequence[Mapping[str, object]] | pd.DataFrame,
    output_dir: str | Path,
    timeout_ms: float = 5000.0,
    instance_group_cols: Sequence[str] = DEFAULT_INSTANCE_GROUP_COLUMNS,
    condition_group_cols: Sequence[str] = DEFAULT_CONDITION_GROUP_COLUMNS,
    stem: str = "experiment",
) -> tuple[Path, Path, Path, Path]:
    """
    Save:
        1) normalized run-level CSV
        2) instance-level CSV
        3) condition-level CSV
        4) summary JSON
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    prepared = prepare_experiment_dataframe(results, timeout_ms=timeout_ms)
    instance_df, condition_df = build_paper_tables(
        results=prepared,
        timeout_ms=timeout_ms,
        instance_group_cols=instance_group_cols,
        condition_group_cols=condition_group_cols,
    )

    runs_csv = output / f"{stem}_runs.csv"
    instance_csv = output / f"{stem}_instance_summary.csv"
    condition_csv = output / f"{stem}_condition_summary.csv"
    summary_json = output / f"{stem}_summary.json"

    prepared.to_csv(runs_csv, index=False)
    instance_df.to_csv(instance_csv, index=False)
    condition_df.to_csv(condition_csv, index=False)

    summary = {
        "timeout_ms": float(timeout_ms),
        "n_runs": int(len(prepared)),
        "n_instance_groups": int(len(instance_df)),
        "n_condition_groups": int(len(condition_df)),
        "instance_group_cols": list(instance_group_cols),
        "condition_group_cols": list(condition_group_cols),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return runs_csv, instance_csv, condition_csv, summary_json