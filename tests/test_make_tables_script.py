from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.make_tables as make_tables


def test_make_tables_main_builds_all_outputs(tmp_path) -> None:
    input_csv = tmp_path / "raw_runs.csv"
    output_dir = tmp_path / "tables"

    runs = pd.DataFrame(
        [
            {
                "method": "CR-PR",
                "env_name": "Narrow",
                "instance_id": 0,
                "success": True,
                "t_init_ms": 100.0,
                "t_res_ms": 1200.0,
                "t_e2e_ms": 1300.0,
                "phi": 0.0,
                "soc": 10.0,
                "soc_ratio": 1.10,
                "makespan": 4.5,
                "n_agents": 4,
                "seed": 0,
            },
            {
                "method": "CR-PR",
                "env_name": "Narrow",
                "instance_id": 0,
                "success": False,
                "t_init_ms": 100.0,
                "t_res_ms": 7000.0,
                "t_e2e_ms": 7100.0,
                "phi": 0.8,
                "soc": 12.0,
                "soc_ratio": 1.50,
                "makespan": 7.0,
                "n_agents": 4,
                "seed": 1,
            },
            {
                "method": "CR-PR",
                "env_name": "Narrow",
                "instance_id": 1,
                "success": True,
                "t_init_ms": 100.0,
                "t_res_ms": 1400.0,
                "t_e2e_ms": 1500.0,
                "phi": 0.1,
                "soc": 10.5,
                "soc_ratio": 1.08,
                "makespan": 4.2,
                "n_agents": 4,
                "seed": 0,
            },
        ]
    )
    runs.to_csv(input_csv, index=False)

    parser = make_tables.build_argument_parser()

    import sys
    argv_backup = sys.argv
    try:
        sys.argv = [
            "make_tables.py",
            "--input-csv", str(input_csv),
            "--output-dir", str(output_dir),
            "--stem", "crpr",
        ]
        make_tables.main()
    finally:
        sys.argv = argv_backup

    prepared_csv = output_dir / "crpr_prepared_runs.csv"
    instance_csv = output_dir / "crpr_instance_summary.csv"
    condition_csv = output_dir / "crpr_condition_summary.csv"
    paper_csv = output_dir / "crpr_paper_table.csv"
    paper_md = output_dir / "crpr_paper_table.md"

    assert prepared_csv.exists()
    assert instance_csv.exists()
    assert condition_csv.exists()
    assert paper_csv.exists()
    assert paper_md.exists()

    prepared_df = pd.read_csv(prepared_csv)
    instance_df = pd.read_csv(instance_csv)
    condition_df = pd.read_csv(condition_csv)
    paper_df = pd.read_csv(paper_csv)
    paper_md_text = paper_md.read_text(encoding="utf-8")

    assert "t_res_ms_capped" in prepared_df.columns
    assert "success_rate" in instance_df.columns
    assert "mean_success_rate" in condition_df.columns

    assert {"method", "env_name", "n_agents", "success_rate"}.issubset(paper_df.columns)
    assert "±" in paper_md_text or "nan" in paper_md_text