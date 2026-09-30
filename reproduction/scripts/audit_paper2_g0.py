#!/usr/bin/env python3
"""Run the initial Paper 2 panel, timing, factor, and test-asset audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq


FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_lines(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one file in {path}, found {len(members)}")
        return archive.read(members[0]).decode("utf-8-sig", errors="strict").splitlines()


def read_factor_file(path: Path, expected: list[str], date_width: int) -> tuple[list[str], dict[str, np.ndarray]]:
    lines = zip_lines(path)
    header_index = next(
        index
        for index, line in enumerate(lines)
        if (lambda fields: fields and fields[0] == "" and all(name in fields[1:] for name in expected))(
            [value.strip() for value in next(csv.reader([line]))]
        )
    )
    header = [value.strip() for value in next(csv.reader([lines[header_index]]))[1:]]
    if header != expected:
        raise ValueError(f"Factor header mismatch in {path}: {header}")
    result: dict[str, np.ndarray] = {}
    for line in lines[header_index + 1 :]:
        fields = next(csv.reader([line])) if line else []
        key = fields[0].strip() if fields else ""
        if len(key) != date_width or not key.isdigit():
            if result:
                break
            continue
        if key in result:
            raise ValueError(f"Duplicate factor date {key} in {path}")
        values = np.asarray([float(value.strip()) / 100.0 for value in fields[1:]], dtype=np.float64)
        if values.size != len(expected) or not np.isfinite(values).all():
            raise ValueError(f"Invalid factor row {key} in {path}")
        result[key] = values
    return header, result


def month_range(start: str, end: str) -> list[str]:
    values = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "M"), dtype="datetime64[M]")
    return [str(value).replace("-", "") for value in values]


def factor_alignment(
    ff5_daily: dict[str, np.ndarray],
    mom_daily: dict[str, np.ndarray],
    ff5_monthly: dict[str, np.ndarray],
    mom_monthly: dict[str, np.ndarray],
    months: list[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    ff5_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    mom_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for day, values in ff5_daily.items():
        ff5_groups[day[:6]].append(values)
    for day, values in mom_daily.items():
        mom_groups[day[:6]].append(values)
    rows: list[dict[str, object]] = []
    for month in months:
        if month not in ff5_monthly or month not in mom_monthly or month not in ff5_groups or month not in mom_groups:
            continue
        daily_sum = np.sum(ff5_groups[month], axis=0)
        mom_sum = float(np.sum(mom_groups[month], axis=0)[0])
        row: dict[str, object] = {
            "month": f"{month[:4]}-{month[4:]}",
            "ff5_trading_days": len(ff5_groups[month]),
            "momentum_trading_days": len(mom_groups[month]),
        }
        for index, name in enumerate(FF5):
            row[f"{name}_daily_sum"] = float(daily_sum[index])
            row[f"{name}_official_monthly"] = float(ff5_monthly[month][index])
            row[f"{name}_difference"] = float(daily_sum[index] - ff5_monthly[month][index])
        row["Mom_daily_sum"] = mom_sum
        row["Mom_official_monthly"] = float(mom_monthly[month][0])
        row["Mom_difference"] = mom_sum - float(mom_monthly[month][0])
        rows.append(row)
    diagnostics: dict[str, dict[str, float]] = {}
    for name in FF5 + ["Mom"]:
        errors = np.abs(np.asarray([float(row[f"{name}_difference"]) for row in rows]))
        diagnostics[name] = {
            "median_absolute_difference": float(np.median(errors)),
            "p95_absolute_difference": float(np.quantile(errors, 0.95)),
            "maximum_absolute_difference": float(np.max(errors)),
        }
    return rows, diagnostics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.config.resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {key: root / config[key] for key in [
        "panel", "pricing_targets", "ff5_monthly", "momentum_monthly", "ff5_daily", "momentum_daily"
    ]}
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=False)

    panel_schema = pl.scan_parquet(paths["panel"]).collect_schema()
    feature_columns = [name for name in panel_schema.names() if name.startswith("x_")]
    missing_columns = [name for name in panel_schema.names() if name.startswith("missing_")]
    panel_stats = (
        pl.scan_parquet(paths["panel"])
        .select(
            pl.len().alias("rows"),
            pl.struct("permno", "month").n_unique().alias("unique_keys"),
            pl.col("permno").n_unique().alias("securities"),
            pl.col("month").min().alias("minimum_feature_month"),
            pl.col("month").max().alias("maximum_feature_month"),
            pl.col("target_month").min().alias("minimum_target_month"),
            pl.col("target_month").max().alias("maximum_target_month"),
            (pl.col("target_month") != pl.col("month").dt.offset_by("1mo")).sum().alias("timing_mismatches"),
            pl.col("ret_fwd1").is_null().sum().alias("missing_forward_returns"),
        )
        .collect()
        .row(0, named=True)
    )
    panel_stats = {key: (str(value) if hasattr(value, "isoformat") else int(value)) for key, value in panel_stats.items()}
    panel_stats["missing_forward_return_fraction"] = panel_stats["missing_forward_returns"] / panel_stats["rows"]

    _, ff5_m = read_factor_file(paths["ff5_monthly"], FF5, 6)
    _, mom_m = read_factor_file(paths["momentum_monthly"], ["Mom"], 6)
    _, ff5_d = read_factor_file(paths["ff5_daily"], FF5, 8)
    _, mom_d = read_factor_file(paths["momentum_daily"], ["Mom"], 8)
    months = month_range(config["development_start"], config["development_end"])
    alignment_rows, factor_diagnostics = factor_alignment(ff5_d, mom_d, ff5_m, mom_m, months)
    write_csv(output / "daily_monthly_factor_alignment.csv", alignment_rows)

    pricing = pq.ParquetFile(paths["pricing_targets"])
    pricing_schema = pricing.schema_arrow
    pricing_months = pq.read_table(paths["pricing_targets"], columns=["month"]).column("month").to_numpy().astype("datetime64[M]")
    required = set(months)
    factor_coverage = set(ff5_m) & set(mom_m) & {key[:6] for key in ff5_d} & {key[:6] for key in mom_d}
    checks = {
        "panel_has_86_features": len(feature_columns) == 86,
        "panel_has_matching_missing_flags": len(missing_columns) == 86,
        "panel_keys_unique": panel_stats["rows"] == panel_stats["unique_keys"],
        "panel_target_is_next_month": panel_stats["timing_mismatches"] == 0,
        "panel_forward_return_missingness_bounded": panel_stats["missing_forward_return_fraction"]
        <= float(config["maximum_forward_return_missing_fraction"]),
        "panel_development_boundary_is_pre2020": panel_stats["maximum_target_month"] == "2019-12-01",
        "six_factor_development_coverage_complete": required.issubset(factor_coverage),
        "factor_daily_calendars_match": set(ff5_d) == set(mom_d).intersection(set(ff5_d)),
        "pricing_target_count_is_74": len(pricing_schema.names) - 1 == 74,
        "pricing_target_months_cover_development": len(pricing_months) == len(months)
        and str(pricing_months[0]) == config["development_start"]
        and str(pricing_months[-1]) == config["development_end"],
    }
    open_items = [
        "Build and audit the stock-day panel used to validate conditional betas; the current Core-86 payload is monthly.",
        "Build confirmation-period fixed test assets after the protocol is frozen; the current public target file ends in 2019-12.",
        "Freeze factor aggregation semantics: daily factors validate beta, official monthly factors define premia; their returns are not assumed algebraically identical.",
        "Freeze the eligible-observation rule: rows with unavailable next-month returns are excluded from supervised losses and reported in coverage tables.",
    ]
    inputs = {name: {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in paths.items()}
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "in_progress" if open_items else "completed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_failed": sum(not value for value in checks.values()),
        "panel": {**panel_stats, "feature_columns": len(feature_columns), "missing_flag_columns": len(missing_columns)},
        "factors": {
            "ff5_daily_rows": len(ff5_d),
            "momentum_daily_rows": len(mom_d),
            "development_months_compared": len(alignment_rows),
            "daily_sum_vs_official_monthly": factor_diagnostics,
            "interpretation": "Differences are diagnostics, not an equality gate: the daily and monthly libraries are distinct horizon constructions."
        },
        "pricing_targets": {"rows": pricing.metadata.num_rows, "assets": len(pricing_schema.names) - 1},
        "open_items": open_items,
        "inputs": inputs,
    }
    (output / "result_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "outputs": [
            {"path": name, "sha256": sha256(output / name)}
            for name in ["daily_monthly_factor_alignment.csv", "environment.json", "result_summary.json"]
        ],
    }
    (output / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
