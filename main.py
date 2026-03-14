from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.config import CRPRConfig
from src.cr_pr import CRPR
from src.environment import Environment
from src.experiment_stats import save_experiment_statistics
from src.metrics import save_run_statistics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("cr_pr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CR-PR continuous conflict-resolution framework")
    parser.add_argument("--env", default="Narrow", choices=["Narrow", "Office", "Warehouse", "C.W."])
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per instance.")
    parser.add_argument("--instances", type=int, default=1, help="Number of instances to run.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--timeout_ms", type=float, default=5000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = CRPRConfig.from_mapping(cfg_data)

    env = Environment.build(args.env)

    results: list[dict] = []
    for instance_id in range(args.instances):
        for seed in range(args.seeds):
            solver = CRPR(
                env=env,
                n_agents=args.n_agents,
                cfg=cfg,
                seed=seed,
                instance_id=instance_id,
            )
            result = solver.run()
            results.append(result)

            logger.info(
                "instance=%02d seed=%02d success=%s t_res=%.1fms t_e2e=%.1fms soc_ratio=%.3f makespan=%.2fs",
                instance_id,
                seed,
                result["success"],
                result["t_res_ms"],
                result["t_e2e_ms"],
                result["soc_ratio"],
                result["makespan"],
            )

    env_stem = env.name.replace(".", "").replace(" ", "_")
    stem = f"{env_stem}_N{args.n_agents}"

    raw_runs_csv = output_dir / f"{stem}_raw_runs.csv"
    pd.DataFrame(results).to_csv(raw_runs_csv, index=False)

    # Backward-compatible single-table summary
    csv_path, json_path = save_run_statistics(results, output_dir, env_stem, args.n_agents)

    # Experiment-oriented multi-level summaries
    runs_csv, instance_csv, condition_csv, summary_json = save_experiment_statistics(
        results=results,
        output_dir=output_dir,
        timeout_ms=args.timeout_ms,
        stem=stem,
    )

    manifest = {
        "config_path": str(config_path),
        "env": env.name,
        "n_agents": args.n_agents,
        "instances": args.instances,
        "seeds_per_instance": args.seeds,
        "timeout_ms": args.timeout_ms,
        "raw_runs_csv": str(raw_runs_csv),
        "legacy_runs_csv": str(csv_path),
        "legacy_summary_json": str(json_path),
        "prepared_runs_csv": str(runs_csv),
        "instance_summary_csv": str(instance_csv),
        "condition_summary_csv": str(condition_csv),
        "experiment_summary_json": str(summary_json),
    }
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Saved raw runs to %s", raw_runs_csv)
    logger.info("Saved legacy run table to %s", csv_path)
    logger.info("Saved legacy summary to %s", json_path)
    logger.info("Saved prepared runs to %s", runs_csv)
    logger.info("Saved instance summary to %s", instance_csv)
    logger.info("Saved condition summary to %s", condition_csv)
    logger.info("Saved experiment summary to %s", summary_json)
    logger.info("Saved manifest to %s", manifest_path)


if __name__ == "__main__":
    main()