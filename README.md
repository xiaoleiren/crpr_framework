# CR-PR Framework

A pure Python + NumPy/SciPy implementation of **CR-PR (Collision Repair via Path Relinking)** for continuous multi-agent path planning in 2D environments.

This repository includes:

- independent single-agent RRT* initialization
- arc-length-based time alignment to a common discrete horizon
- diversity-preserving per-agent elite archives
- linear path relinking between elite trajectories
- kinematic repair operator `L_beta`
- swept collision checking with sparse broadphase filtering
- lexicographic objective `(Phi, SoC)`
- experiment aggregation utilities for instance-level and condition-level reporting
- benchmark scripts for reproducible runs and paper-style tables


## Repository Structure

```
cr-pr/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ config.yaml
├─ main.py
├─ scripts/
│  ├─ run_benchmark.py
│  └─ make_tables.py
├─ src/
│  ├─ __init__.py
│  ├─ collision.py
│  ├─ config.py
│  ├─ cr_pr.py
│  ├─ elite_manager.py
│  ├─ environment.py
│  ├─ experiment_stats.py
│  ├─ geometry.py
│  ├─ kinematic.py
│  ├─ metrics.py
│  ├─ objective.py
│  ├─ repair_operator.py
│  └─ rrt_star.py
├─ tests/
│  ├─ conftest.py
│  ├─ test_collision.py
│  ├─ test_crpr_integration.py
│  ├─ test_crpr_metadata.py
│  ├─ test_crpr_smoke.py
│  ├─ test_environment.py
│  ├─ test_experiment_stats.py
│  ├─ test_geometry.py
│  ├─ test_kinematic.py
│  ├─ test_metrics.py
│  ├─ test_objective.py
│  ├─ test_repair_operator.py
│  └─ test_rrt_star.py
└─ results/
```

## Core Features

### Planning pipeline

- independent RRT* planning for each agent
- path resampling / time alignment to a common time horizon
- elite archive construction for each agent
- pair selection from elite archives
- linear relinking over `alpha in {0, Δα, ..., 1}`
- collision repair using `L_beta`
- acceptance under lexicographic score improvement

### Feasibility and safety

- obstacle and boundary feasibility checks
- swept segment-segment collision checking between agents
- swept segment-rectangle checking for obstacles
- kinematic feasibility under velocity and acceleration bounds
- relinking-segment clearance screening heuristic

### Evaluation and reporting

- run-level solver outputs with experiment metadata
- instance-level aggregation over seeds
- condition-level aggregation over instances
- capped runtime summaries
- success-only aggregation for `soc_ratio`, `makespan`, and `phi`
- paired-test and bootstrap utilities in `src/metrics.py`

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

## Optional editable install:
```
pip install -e .
```

## Quick Start

Run the full test suite:
```

pytest tests -q
```

Run a simple solver entrypoint:
```
python main.py --env Narrow --n_agents 20 --seeds 5
```

Running Benchmarks

Use the benchmark script to generate raw run-level outputs and aggregated statistics.

Example:
```
python scripts/run_benchmark.py \
  --envs Narrow,Office,Warehouse \
  --n-agents 4,8,16 \
  --instances 10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --output-dir outputs/benchmark \
  --stem crpr
```
## This script saves:

raw run-level CSV

prepared run-level CSV with capped runtimes

instance-level summary CSV

condition-level summary CSV

summary JSON

manifest JSON

# Building Paper-Style Tables

Generate paper-ready tables from a raw or prepared runs CSV:
```
python scripts/make_tables.py \
  --input-csv outputs/benchmark/crpr_raw_runs.csv \
  --output-dir outputs/tables \
  --stem crpr
```
# This script saves:

prepared runs CSV

instance summary CSV

condition summary CSV

paper-style CSV table

paper-style Markdown table

## Output Fields

A typical CRPR.run() result dictionary includes:

method

env_name

instance_id

success

t_init_ms

t_res_ms

t_e2e_ms

phi

soc

soc_ratio

makespan

n_agents

seed

accepted_candidates

candidate_count

mean_repair_passes

outer_iterations

These fields are directly consumed by src/experiment_stats.py.

## Aggregation Logic

src/experiment_stats.py performs three levels of processing:

run-level normalization

instance-level aggregation over seeds

condition-level aggregation over instances

Default grouping:

instance-level: (method, env_name, n_agents, instance_id)

condition-level: (method, env_name, n_agents)

Runtime statistics are capped by timeout_ms.

Metrics such as soc_ratio, makespan, and phi are aggregated over successful runs only.

## Supported Environments

Currently included environments:

Narrow

Office

Warehouse

These are defined in src/environment.py.

## Important Note

This repository keeps the implementation in pure Python + NumPy/SciPy.

The swept collision oracle is therefore a geometric approximation rather than a literal FCL-backed implementation. If exact parity with an FCL-based checker is required, src/collision.py can be replaced with a python-fcl backed implementation while preserving the same public interface.

Likewise, the current implementation follows the paper-style CR-PR workflow closely, while some engineering choices remain approximation-oriented for portability and reproducibility.

## Reproducibility Notes

start/goal sampling is seed-controlled

solver outputs include seed, env_name, and instance_id

benchmark runs save a manifest JSON with configuration and run ranges

outputs are stored in CSV/JSON format for downstream analysis

## Recommended Workflow

run pytest to validate the repository

run scripts/run_benchmark.py for experiment generation

inspect raw and prepared run-level CSVs

run scripts/make_tables.py to build summary tables

export the resulting CSV / Markdown tables into reports or papers

### Result files produced by `main.py`

A typical run such as:

```
python main.py --env Narrow --n_agents 2 --seeds 5 --instances 1 --output results
```
produces files like:

Narrow_N2_raw_runs.csv: raw run-level outputs from CRPR.run()

Narrow_N2_legacy_runs.csv: backward-compatible single-table summary input

Narrow_N2_legacy_summary.json: backward-compatible aggregate summary

Narrow_N2_runs.csv: normalized run-level table with capped runtimes

Narrow_N2_instance_summary.csv: aggregation over seeds for each instance

Narrow_N2_condition_summary.csv: aggregation over instances for each condition

Narrow_N2_experiment_summary.json: experiment-level summary metadata

Narrow_N2_manifest.json: run manifest with configuration and output paths

## Using `config.yaml` and `main.py`

The file `config.yaml` stores the default solver hyperparameters used to build `CRPRConfig`.

Run a single experiment batch with:

```
python main.py --env Narrow --n_agents 20 --seeds 5 --instances 1 --config config.yaml --output results
```

Arguments:

--env: one of Narrow, Office, Warehouse

--n_agents: number of agents

--seeds: number of seeds per instance

--instances: number of instances to run

--config: path to YAML config file

--output: directory for saving results

This command saves:

raw run-level CSV

legacy summary CSV / JSON

prepared run-level CSV

instance-level summary CSV

condition-level summary CSV

manifest JSON



