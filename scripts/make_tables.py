from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_stats import build_paper_tables, prepare_experiment_dataframe


def _format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    if pd.isna(mean):
        return "nan"
    if pd.isna(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _load_runs_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


def _paper_condition_table(condition_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in condition_df.iterrows():
        paper_row = {
            "method": row["method"],
            "env_name": row["env_name"],
            "n_agents": int(row["n_agents"]),
            "n_instances": int(row["n_instances"]),
            "success_rate": _format_mean_std(row["mean_success_rate"], row["std_success_rate"], precision=3),
            "runtime_res_ms": _format_mean_std(row["mean_runtime_res_ms"], row["std_runtime_res_ms"], precision=1),
            "runtime_e2e_ms": _format_mean_std(row["mean_runtime_e2e_ms"], row["std_runtime_e2e_ms"], precision=1),
            "soc_ratio_success": _format_mean_std(
                row["mean_soc_ratio_success"],
                row["std_soc_ratio_success"],
                precision=3,
            ),
            "makespan_success": _format_mean_std(
                row["mean_makespan_success"],
                row["std_makespan_success"],
                precision=3,
            ),
        }

        if "mean_phi_success" in condition_df.columns:
            paper_row["phi_success"] = _format_mean_std(
                row.get("mean_phi_success", np.nan),
                row.get("std_phi_success", np.nan),
                precision=3,
            )

        if "timeout_ms" in condition_df.columns:
            paper_row["timeout_ms"] = float(row["timeout_ms"])

        rows.append(paper_row)

    return pd.DataFrame(rows)


def _save_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.write_text(df.to_markdown(index=False), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper tables from CR-PR run CSV.")

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to raw runs CSV or prepared runs CSV.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=float,
        default=5000.0,
        help="Runtime cap used for statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tables",
        help="Directory to save generated tables.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="crpr_tables",
        help="Filename stem for outputs.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.timeout_ms <= 0.0:
        raise ValueError("--timeout-ms must be positive.")

    runs_df = _load_runs_csv(args.input_csv)
    prepared_df = prepare_experiment_dataframe(runs_df, timeout_ms=args.timeout_ms)
    instance_df, condition_df = build_paper_tables(prepared_df, timeout_ms=args.timeout_ms)
    paper_df = _paper_condition_table(condition_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_csv = output_dir / f"{args.stem}_prepared_runs.csv"
    instance_csv = output_dir / f"{args.stem}_instance_summary.csv"
    condition_csv = output_dir / f"{args.stem}_condition_summary.csv"
    paper_csv = output_dir / f"{args.stem}_paper_table.csv"
    paper_md = output_dir / f"{args.stem}_paper_table.md"

    prepared_df.to_csv(prepared_csv, index=False)
    instance_df.to_csv(instance_csv, index=False)
    condition_df.to_csv(condition_csv, index=False)
    paper_df.to_csv(paper_csv, index=False)
    _save_markdown_table(paper_df, paper_md)

    print(f"Saved prepared runs to: {prepared_csv}")
    print(f"Saved instance summary to: {instance_csv}")
    print(f"Saved condition summary to: {condition_csv}")
    print(f"Saved paper CSV table to: {paper_csv}")
    print(f"Saved paper Markdown table to: {paper_md}")


if __name__ == "__main__":
    main()