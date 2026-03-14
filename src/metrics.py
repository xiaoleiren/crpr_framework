from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, bootstrap, norm, wilcoxon


def _ensure_same_length(a: Sequence[object], b: Sequence[object], name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name_a} and {name_b} must have the same length.")


def summarize_results(results: Sequence[Mapping[str, float | int | bool]]) -> pd.DataFrame:
    return pd.DataFrame(results)


def mcnemar_test(success_a: Sequence[bool], success_b: Sequence[bool]) -> dict[str, float | int | None]:
    _ensure_same_length(success_a, success_b, "success_a", "success_b")

    a = np.asarray(success_a, dtype=bool)
    b = np.asarray(success_b, dtype=bool)

    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10

    p_value = 1.0 if n == 0 else float(
        binomtest(k=min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue
    )

    odds_ratio = None
    if b10 > 0:
        odds_ratio = float(b01 / b10)

    return {
        "discordant_a_only": b01,
        "discordant_b_only": b10,
        "n_discordant": n,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
    }


def paired_runtime_reduction(crpr_times: Sequence[float], baseline_times: Sequence[float]) -> np.ndarray:
    _ensure_same_length(crpr_times, baseline_times, "crpr_times", "baseline_times")

    crpr = np.asarray(crpr_times, dtype=float)
    base = np.asarray(baseline_times, dtype=float)
    return 1.0 - crpr / np.maximum(base, 1e-9)


def wilcoxon_runtime_test(reductions: Sequence[float]) -> dict[str, float | int]:
    reductions = np.asarray(reductions, dtype=float)
    mask = np.abs(reductions) > 1e-12
    filtered = reductions[mask]

    if filtered.size == 0:
        return {"n": 0, "p_value": 1.0, "effect_r": 0.0}

    result = wilcoxon(filtered, zero_method="wilcox", alternative="two-sided", correction=False)
    z_abs = abs(norm.isf(result.pvalue / 2.0)) if result.pvalue > 0.0 else np.inf
    effect_r = float(z_abs / np.sqrt(filtered.size))

    return {
        "n": int(filtered.size),
        "p_value": float(result.pvalue),
        "effect_r": effect_r,
    }


def bca_confidence_interval(
    samples: Sequence[float],
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
) -> dict[str, float]:
    data = np.asarray(samples, dtype=float)
    if data.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")

    res = bootstrap(
        data=(data,),
        statistic=np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="BCa",
        random_state=123,
    )

    return {
        "mean": float(np.mean(data)),
        "low": float(res.confidence_interval.low),
        "high": float(res.confidence_interval.high),
    }


def save_run_statistics(
    results: Sequence[Mapping[str, float | int | bool]],
    output_dir: str | Path,
    env_name: str,
    n_agents: int,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = summarize_results(results)
    csv_path = output / f"{env_name}_N{n_agents}.csv"
    json_path = output / f"{env_name}_N{n_agents}_summary.json"

    df.to_csv(csv_path, index=False)

    summary = {
        "success_rate": float(df["success"].mean()),
        "mean_t_res_ms": float(df["t_res_ms"].mean()),
        "mean_t_e2e_ms": float(df["t_e2e_ms"].mean()),
        "mean_soc_ratio": float(df["soc_ratio"].mean()),
        "mean_makespan": float(df["makespan"].mean()),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path