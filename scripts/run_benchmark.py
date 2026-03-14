from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CRPRConfig
from src.cr_pr import CRPR
from src.environment import Environment
from src.experiment_stats import save_experiment_statistics


DEFAULT_CONFIG: dict[str, Any] = {
    "delta_alpha": 0.1,
    "beta": 0.5,
    "K": 3,
    "elite_max": 4,
    "d_max": 0.05,
    "epsilon": 0.001,
    "v_max": 1.0,
    "a_max": 2.0,
    "r_agent": 0.3,
    "dt": 0.05,
    "timeout_s": 5.0,
    "rrt_max_iter": 400,
    "rrt_step_size": 0.8,
    "rrt_goal_bias": 0.2,
    "rrt_neighbor_radius": 1.5,
    "min_clearance_skip": 0.02,
    "init_noise_sigma": 0.02,
    "cache_refresh_every": 2,
    "ablation": "full",
    "seed": 0,
}


def _parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of integers.")
    return values


def _parse_str_list(text: str) -> list[str]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(part)
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of strings.")
    return values


def _load_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_CONFIG)

    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(data)
    return cfg


def _run_single(
    env_name: str,
    n_agents: int,
    instance_id: int,
    seed: int,
    cfg_dict: dict[str, Any],
) -> dict[str, Any]:
    cfg_payload = dict(cfg_dict)
    cfg_payload["seed"] = seed
    cfg = CRPRConfig.from_mapping(cfg_payload)

    env = Environment.build(env_name)
    solver = CRPR(
        env=env,
        n_agents=n_agents,
        cfg=cfg,
        seed=seed,
        instance_id=instance_id,
    )
    result = solver.run()
    return dict(result)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CR-PR benchmark experiments.")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file for CRPRConfig. Defaults to built-in config.",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default="Narrow,Office,Warehouse",
        help="Comma-separated environment names.",
    )
    parser.add_argument(
        "--n-agents",
        type=str,
        default="4,8,16",
        help="Comma-separated agent counts.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=10,
        help="Number of instances per (env, n_agents).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
        help="Comma-separated random seeds per instance.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=float,
        default=5000.0,
        help="Runtime cap used by experiment statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="crpr",
        help="Filename stem for saved outputs.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    env_names = _parse_str_list(args.envs)
    agent_counts = _parse_int_list(args.n_agents)
    seeds = _parse_int_list(args.seeds)

    if args.instances < 1:
        raise ValueError("--instances must be >= 1.")
    if args.timeout_ms <= 0.0:
        raise ValueError("--timeout-ms must be positive.")

    cfg_dict = _load_config(args.config)

    all_results: list[dict[str, Any]] = []
    total_runs = len(env_names) * len(agent_counts) * args.instances * len(seeds)
    run_index = 0

    for env_name in env_names:
        for n_agents in agent_counts:
            for instance_id in range(args.instances):
                for seed in seeds:
                    run_index += 1
                    print(
                        f"[{run_index}/{total_runs}] "
                        f"env={env_name} n_agents={n_agents} instance={instance_id} seed={seed}",
                        flush=True,
                    )
                    result = _run_single(
                        env_name=env_name,
                        n_agents=n_agents,
                        instance_id=instance_id,
                        seed=seed,
                        cfg_dict=cfg_dict,
                    )
                    all_results.append(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_runs_csv = output_dir / f"{args.stem}_raw_runs.csv"
    pd.DataFrame(all_results).to_csv(raw_runs_csv, index=False)

    runs_csv, instance_csv, condition_csv, summary_json = save_experiment_statistics(
        results=all_results,
        output_dir=output_dir,
        timeout_ms=args.timeout_ms,
        stem=args.stem,
    )

    manifest = {
        "config": cfg_dict,
        "envs": env_names,
        "n_agents": agent_counts,
        "instances": int(args.instances),
        "seeds": seeds,
        "timeout_ms": float(args.timeout_ms),
        "raw_runs_csv": str(raw_runs_csv),
        "prepared_runs_csv": str(runs_csv),
        "instance_summary_csv": str(instance_csv),
        "condition_summary_csv": str(condition_csv),
        "summary_json": str(summary_json),
    }
    manifest_path = output_dir / f"{args.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved raw runs to: {raw_runs_csv}")
    print(f"Saved prepared runs to: {runs_csv}")
    print(f"Saved instance summary to: {instance_csv}")
    print(f"Saved condition summary to: {condition_csv}")
    print(f"Saved summary JSON to: {summary_json}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()