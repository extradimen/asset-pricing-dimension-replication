"""Verify tree evidence and explore a common A/B mechanism break across models."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calendar(start, end):
    result = []
    year, month = divmod(start, 100)
    while year * 100 + month <= end:
        result.append(year * 100 + month)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def verify_archive(spec):
    root = Path(spec["dir"])
    manifest_path = root / "output_manifest.json"
    if sha(manifest_path) != spec["manifest_sha256"]:
        raise RuntimeError(f"manifest mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["outputs"].items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"output hash mismatch: {path}")
    monthly = root / spec["monthly_file"]
    if spec["monthly_file"] not in manifest["outputs"]:
        raise RuntimeError(f"monthly file absent from manifest: {monthly}")
    return json.loads(monthly.read_text())


def summarize(rows, arm, mask):
    selected = [row[arm] for row, keep in zip(rows, mask) if keep]
    a = float(np.mean([row["adjustment_cost_A"] for row in selected]))
    two_b = float(np.mean([row["alignment_benefit_2B"] for row in selected]))
    delta = float(np.mean([row["updated_minus_frozen_loss"] for row in selected]))
    return {
        "months": len(selected), "adjustment_cost_A": a, "alignment_benefit_2B": two_b,
        "updated_minus_frozen_loss": delta,
        "ex_post_lambda_star_from_mean_terms": float(np.clip((two_b / 2) / a, 0, 1)) if a > 0 else 0.0,
        "months_full_update_helped": int(sum(row["updated_minus_frozen_loss"] < 0 for row in selected)),
    }


def standardize(matrix):
    scale = matrix.std(axis=0, ddof=1)
    if np.any(scale <= 0):
        raise RuntimeError("constant dominance series")
    return (matrix - matrix.mean(axis=0)) / scale, scale


def break_scores(z, minimum):
    total_mean = z.mean(axis=0)
    sse0 = float(np.sum((z - total_mean) ** 2))
    scores = []
    for split in range(minimum, len(z) - minimum + 1):
        before, after = z[:split], z[split:]
        sse1 = float(np.sum((before - before.mean(axis=0)) ** 2) +
                     np.sum((after - after.mean(axis=0)) ** 2))
        scores.append((split, sse0 - sse1))
    return scores


def circular_blocks(values, block, rng):
    n = len(values)
    starts = rng.integers(0, n, size=(n + block - 1) // block)
    indices = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
    return values[indices]


def best_break(z, minimum):
    scores = break_scores(z, minimum)
    return max(scores, key=lambda item: item[1]), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    expected_months = calendar(cfg["evaluation_start"], cfg["evaluation_end"])
    rows_by_model = {name: verify_archive(spec) for name, spec in cfg["inputs"].items()}
    for model, rows in rows_by_model.items():
        months = [row["month"] for row in rows]
        if months != expected_months:
            raise RuntimeError(f"calendar mismatch: {model}")
        for row in rows:
            for arm in cfg["arms"]:
                part = row[arm]
                reconstructed = part["adjustment_cost_A"] - part["alignment_benefit_2B"]
                if abs(reconstructed - part["updated_minus_frozen_loss"]) > 1e-12:
                    raise RuntimeError(f"identity mismatch: {model}:{row['month']}:{arm}")

    development = np.array([month <= cfg["development_end"] for month in expected_months])
    promotion = np.array([month >= cfg["promotion_start"] for month in expected_months])
    full = np.ones(len(expected_months), dtype=bool)
    summaries = {}
    series, labels = [], []
    for model, rows in rows_by_model.items():
        summaries[model] = {}
        for arm in cfg["arms"]:
            summaries[model][arm] = {
                "full_2000_2019": summarize(rows, arm, full),
                "development_2000_2009": summarize(rows, arm, development),
                "promotion_2010_2019": summarize(rows, arm, promotion),
            }
            a = np.array([row[arm]["adjustment_cost_A"] for row in rows])
            two_b = np.array([row[arm]["alignment_benefit_2B"] for row in rows])
            score = (two_b - a) / (a + np.abs(two_b) + 1e-15)
            if not np.isfinite(score).all() or np.max(np.abs(score)) > 1 + 1e-9:
                raise RuntimeError(f"invalid dominance score: {model}:{arm}")
            series.append(score)
            labels.append(f"{model}:{arm}")
    matrix = np.column_stack(series)
    z, scales = standardize(matrix)
    minimum = cfg["minimum_segment_months"]
    (split, observed), scores = best_break(z, minimum)

    null_cfg = cfg["null_bootstrap"]
    rng = np.random.default_rng(null_cfg["seed"])
    centered = z - z.mean(axis=0)
    null_stats = np.empty(null_cfg["replications"])
    for index in range(len(null_stats)):
        draw = circular_blocks(centered, null_cfg["block_months"], rng)
        null_stats[index] = best_break(draw, minimum)[0][1]
    p_value = float((1 + np.sum(null_stats >= observed)) / (len(null_stats) + 1))

    before_mean, after_mean = z[:split].mean(axis=0), z[split:].mean(axis=0)
    residuals = np.vstack([z[:split] - before_mean, z[split:] - after_mean])
    alt_cfg = cfg["break_date_interval_bootstrap"]
    rng = np.random.default_rng(alt_cfg["seed"])
    locations = np.empty(alt_cfg["replications"], dtype=int)
    fitted = np.vstack([np.tile(before_mean, (split, 1)), np.tile(after_mean, (len(z) - split, 1))])
    for index in range(len(locations)):
        draw = fitted + circular_blocks(residuals, alt_cfg["block_months"], rng)
        locations[index] = best_break(draw, minimum)[0][0]
    low_q, high_q = alt_cfg["percentiles"]
    low = int(np.quantile(locations, low_q / 100, method="nearest"))
    high = int(np.quantile(locations, high_q / 100, method="nearest"))

    report = {
        "experiment_id": cfg["experiment_id"], "scope": cfg["scope"],
        "claim_boundary": cfg["claim_boundary"], "months": len(expected_months),
        "archive_verification": {"manifests": "pass", "all_output_hashes": "pass",
                                 "calendar": "pass", "exact_loss_identities": "pass"},
        "summaries": summaries,
        "common_break": {
            "series": labels, "estimated_first_month_after_break": expected_months[split],
            "last_month_before_break": expected_months[split - 1], "split_index": split,
            "pooled_sse_reduction": observed, "exploratory_null_bootstrap_p_value": p_value,
            "bootstrap_95pct_first_month_after_break_interval": [expected_months[low], expected_months[high]],
            "standardized_mean_shift_by_series": {
                label: float(after - before) for label, before, after in zip(labels, before_mean, after_mean)
            },
            "positive_shift_series_count": int(np.sum(after_mean > before_mean)),
            "bootstrap": {"null_replications": len(null_stats), "alternative_replications": len(locations),
                          "block_months": null_cfg["block_months"]},
            "interpretation": "Retrospective exploratory break only; not a real-time environment definition or confirmatory test."
        }
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    series_rows = []
    for index, month in enumerate(expected_months):
        row = {"month": month}
        for column, label in enumerate(labels):
            row[label] = {"dominance_score": float(matrix[index, column]),
                          "standardized_score": float(z[index, column])}
        series_rows.append(row)
    (out / "dominance_series.json").write_text(json.dumps(series_rows, ensure_ascii=False, indent=2) + "\n")
    profile = [{"first_month_after_break": expected_months[index], "pooled_sse_reduction": value}
               for index, value in scores]
    (out / "break_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "input_manifests": {name: spec["manifest_sha256"] for name, spec in cfg["inputs"].items()},
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "numpy": np.__version__},
        "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    }
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
