"""Evaluate a locked causal common-signal CUSUM update switch."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from statsmodels.stats.multitest import multipletests
import statsmodels


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(spec):
    root = Path(spec["dir"])
    manifest_path = root / "output_manifest.json"
    if sha(manifest_path) != spec["manifest_sha256"]:
        raise RuntimeError(f"manifest mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["outputs"].items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"output mismatch: {path}")
    if spec["monthly_file"] not in manifest["outputs"]:
        raise RuntimeError("monthly input is not archived")
    return json.loads((root / spec["monthly_file"]).read_text())


def circular_indices(n, length, block, rng):
    count = (length + block - 1) // block
    starts = rng.integers(0, n, size=count)
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:length]


def cusum_path(signal, reference):
    values, running = [], 0.0
    for value in signal:
        running = max(0.0, running + value - reference)
        values.append(running)
    return np.asarray(values)


def calibrate_threshold(dev_matrix, monitor_months, reference, block, draws, alpha, seed):
    centered = dev_matrix - dev_matrix.mean(axis=0)
    scale = dev_matrix.std(axis=0, ddof=1)
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    for draw in range(draws):
        index = circular_indices(len(centered), monitor_months, block, rng)
        common = (centered[index] / scale).mean(axis=1)
        maxima[draw] = cusum_path(common, reference).max()
    threshold = float(np.quantile(maxima, 1 - alpha, method="higher"))
    return threshold, maxima


def centered_block_test(values, draws, block, seed):
    values = np.asarray(values, dtype=float)
    estimate = float(values.mean())
    centered = values - estimate
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        index = circular_indices(len(values), len(values), block, rng)
        samples[draw] = centered[index].mean()
    p = float((1 + np.sum(np.abs(samples) >= abs(estimate))) / (draws + 1))
    q = np.quantile(samples, [0.025, 0.975])
    return {"estimate": estimate, "basic_ci95": [float(estimate - q[1]), float(estimate - q[0])],
            "p_two_sided_centered": p}


def leave_one_year(values, months):
    years = sorted(set(months // 100))
    return {str(year): float(values[months // 100 != year].mean()) for year in years}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rows = {name: verify_archive(spec) for name, spec in cfg["inputs"].items()}
    months = np.asarray([row["month"] for row in next(iter(rows.values()))], dtype=int)
    if any([row["month"] for row in value] != months.tolist() for value in rows.values()):
        raise RuntimeError("cross-model calendar mismatch")
    dev = (months >= cfg["development_start"]) & (months <= cfg["development_end"])
    promotion = (months >= cfg["promotion_start"]) & (months <= cfg["promotion_end"])
    if dev.sum() != 119 or promotion.sum() != 120:
        raise RuntimeError("development or promotion calendar mismatch")

    columns, labels = [], []
    for model, model_rows in rows.items():
        for arm in cfg["arms"]:
            a = np.asarray([row[arm]["adjustment_cost_A"] for row in model_rows])
            two_b = np.asarray([row[arm]["alignment_benefit_2B"] for row in model_rows])
            delta = np.asarray([row[arm]["updated_minus_frozen_loss"] for row in model_rows])
            if np.max(np.abs(delta - (a - two_b))) > 1e-12:
                raise RuntimeError(f"identity mismatch: {model}:{arm}")
            columns.append((two_b - a) / (a + np.abs(two_b) + 1e-15))
            labels.append(f"{model}:{arm}")
    matrix = np.column_stack(columns)
    dev_matrix, promotion_matrix = matrix[dev], matrix[promotion]
    means = dev_matrix.mean(axis=0)
    scales = dev_matrix.std(axis=0, ddof=1)
    if np.any(scales <= 0):
        raise RuntimeError("constant development series")
    common_signal = ((promotion_matrix - means) / scales).mean(axis=1)
    detector = cfg["sequential_detector"]
    reference = detector["reference_shift_sigma"]
    threshold, null_maxima = calibrate_threshold(
        dev_matrix, int(promotion.sum()), reference, detector["block_months"],
        detector["bootstrap_replications"], detector["alpha"], detector["seed"])
    path = cusum_path(common_signal, reference)
    crossings = np.flatnonzero(path > threshold)
    crossing_index = int(crossings[0]) if len(crossings) else None
    active = np.zeros(int(promotion.sum()), dtype=bool)
    if crossing_index is not None and crossing_index + 1 < len(active):
        active[crossing_index + 1:] = True
    promotion_months = months[promotion]

    switched = {}
    primary_endpoints, endpoint_order = {}, []
    for model, model_rows in rows.items():
        selected_rows = [row for row, keep in zip(model_rows, promotion) if keep]
        switched[model] = {}
        for arm in cfg["arms"]:
            full_delta = np.asarray([row[arm]["updated_minus_frozen_loss"] for row in selected_rows])
            switch_delta = np.where(active, full_delta, 0.0)
            versus_frozen = -switch_delta
            versus_full = full_delta - switch_delta
            switched[model][arm] = {
                "switched_minus_frozen_loss": float(switch_delta.mean()),
                "full_minus_frozen_loss": float(full_delta.mean()),
                "improvement_vs_frozen": float(versus_frozen.mean()),
                "improvement_vs_full": float(versus_full.mean()),
                "active_months": int(active.sum()),
            }
            if arm == cfg["primary_arm"]:
                for comparison, values in [("vs_frozen", versus_frozen), ("vs_full", versus_full)]:
                    key = f"{model}:{comparison}"
                    primary_endpoints[key] = values
                    endpoint_order.append(key)

    infer_cfg = cfg["inference"]
    inference = {}
    for index, key in enumerate(endpoint_order):
        result = centered_block_test(primary_endpoints[key], infer_cfg["bootstrap_replications"],
                                     infer_cfg["block_months"], infer_cfg["seed"] + index)
        result["leave_one_year_out"] = leave_one_year(primary_endpoints[key], promotion_months)
        inference[key] = result
    adjusted = multipletests([inference[key]["p_two_sided_centered"] for key in endpoint_order],
                             method="holm")[1]
    for key, p_holm in zip(endpoint_order, adjusted):
        inference[key]["p_holm"] = float(p_holm)
        inference[key]["positive_all_leave_one_year_out"] = min(inference[key]["leave_one_year_out"].values()) > 0
        inference[key]["strict_pass"] = (inference[key]["estimate"] > 0 and
            inference[key]["basic_ci95"][0] > 0 and p_holm < infer_cfg["holm_alpha"] and
            inference[key]["positive_all_leave_one_year_out"])
    main_models_pass = all(inference[f"{model}:{comparison}"]["strict_pass"]
                           for model in ["neural", "tree"] for comparison in ["vs_frozen", "vs_full"])
    ridge_nonnegative = all(inference[f"ridge:{comparison}"]["estimate"] >= 0
                            for comparison in ["vs_frozen", "vs_full"])
    gate = bool(main_models_pass and ridge_nonnegative)

    decisions = [{"month": int(month), "common_signal": float(signal), "cusum": float(stat),
                  "threshold": threshold, "active_update": bool(on)}
                 for month, signal, stat, on in zip(promotion_months, common_signal, path, active)]
    report = {
        "experiment_id": cfg["experiment_id"], "scope": cfg["scope"],
        "claim_boundary": cfg["claim_boundary"], "archive_verification": "pass",
        "detector": {"development_months": int(dev.sum()), "monitor_months": int(promotion.sum()),
                     "series": labels, "reference_shift_sigma": reference, "threshold": threshold,
                     "null_maximum_95pct": threshold,
                     "crossing_month": int(promotion_months[crossing_index]) if crossing_index is not None else None,
                     "first_active_month": int(promotion_months[crossing_index + 1])
                     if crossing_index is not None and crossing_index + 1 < len(promotion_months) else None,
                     "active_months": int(active.sum())},
        "switched_results": switched, "primary_inference": inference,
        "statistical_promotion_gate_pass": gate,
        "next_action": ("freeze economic-value gate before any later-period outcome access" if gate else
                        "close this sequential detector without revision; do not access 2020-2025 model outcomes"),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "monthly_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    manifest = {"config_sha256": sha(args.config), "script_sha256": sha(__file__),
                "input_manifests": {name: spec["manifest_sha256"] for name, spec in cfg["inputs"].items()},
                "runtime": {"python": platform.python_version(), "executable": sys.executable,
                            "numpy": np.__version__, "statsmodels": statsmodels.__version__},
                "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()}}
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
