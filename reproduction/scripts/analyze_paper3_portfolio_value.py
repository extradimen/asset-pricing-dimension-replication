"""Transaction-cost-aware portfolios from archived stock-level predictions."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from statsmodels.stats.multitest import multipletests
import statsmodels

from run_paper3_q2_baseline import load_zip


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(root, expected):
    path = Path(root) / "output_manifest.json"
    if sha(path) != expected:
        raise RuntimeError(f"manifest mismatch: {root}")
    return json.loads(path.read_text())


def verify(root, manifest, relative):
    path = Path(root) / relative
    if relative not in manifest["outputs"] or sha(path) != manifest["outputs"][relative]:
        raise RuntimeError(f"archived file mismatch: {path}")
    return path


def target_weights(prediction, market_cap, deciles, weighting):
    n = len(prediction)
    size = n // deciles
    if size < 1 or not np.isfinite(prediction).all():
        raise RuntimeError("invalid portfolio cross section")
    order = np.argsort(prediction, kind="stable")
    bottom, top = order[:size], order[-size:]
    result = np.zeros(n, dtype=np.float64)
    if weighting == "equal":
        result[top], result[bottom] = 1 / len(top), -1 / len(bottom)
    elif weighting == "value":
        for index, sign in [(top, 1.0), (bottom, -1.0)]:
            scaled = np.exp(market_cap[index] - np.max(market_cap[index]))
            result[index] = sign * scaled / scaled.sum()
    else:
        raise ValueError(weighting)
    return result


def drift_weights(previous, previous_total_return):
    result = np.zeros_like(previous)
    long = previous > 0
    short = previous < 0
    if long.any():
        gross = 1 + float(np.sum(previous[long] * previous_total_return[long]))
        result[long] = previous[long] * (1 + previous_total_return[long]) / gross
    if short.any():
        magnitude = -previous[short]
        gross = 1 + float(np.sum(magnitude * previous_total_return[short]))
        result[short] = -(magnitude * (1 + previous_total_return[short]) / gross)
    return result


def turnover(current_ids, current_weights, previous):
    if previous is None:
        return 1.0
    previous_ids, previous_weights, previous_total_return = previous
    drifted = drift_weights(previous_weights, previous_total_return)
    old = {int(key): float(value) for key, value in zip(previous_ids, drifted)}
    new = {int(key): float(value) for key, value in zip(current_ids, current_weights)}
    keys = set(old) | set(new)
    return 0.5 * sum(abs(new.get(key, 0.0) - old.get(key, 0.0)) for key in keys)


def circular_indices(n, block, rng):
    count = (n + block - 1) // block
    starts = rng.integers(0, n, size=count)
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]


def block_test(values, draws, block, seed):
    values = np.asarray(values, dtype=float)
    estimate = float(values.mean())
    centered = values - estimate
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        samples[draw] = centered[circular_indices(len(values), block, rng)].mean()
    q = np.quantile(samples, [0.025, 0.975])
    return {"estimate": estimate, "basic_ci95": [float(estimate-q[1]), float(estimate-q[0])],
            "p_two_sided_centered": float((1 + np.sum(np.abs(samples) >= abs(estimate))) / (draws + 1))}


def performance(rows, gamma):
    values = np.asarray([row["net_return"] for row in rows])
    mean, volatility = float(values.mean()), float(values.std(ddof=1))
    return {"months": len(rows), "annualized_mean": 12 * mean,
            "annualized_volatility": np.sqrt(12) * volatility,
            "annualized_sharpe": np.sqrt(12) * mean / volatility if volatility > 0 else None,
            "annualized_certainty_equivalent_gamma5": 12 * (mean - gamma * values.var(ddof=1) / 2),
            "mean_turnover": float(np.mean([row["turnover"] for row in rows])),
            "positive_month_fraction": float(np.mean(values > 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if sha(cfg["preprocessing"]) != cfg["preprocessing_sha256"] or sha(cfg["rf_zip"]) != cfg["rf_sha256"]:
        raise RuntimeError("external input hash mismatch")
    meta = json.loads(Path(cfg["preprocessing"]).read_text())
    if meta["continuous"][0] != "log_market_cap":
        raise RuntimeError("market-cap feature position mismatch")
    cap_mean, cap_scale = meta["training_mean"][0], meta["training_scale"][0]
    roots = {"frozen": cfg["frozen_dir"], "updated": cfg["neural_updated_dir"],
             "tree": cfg["tree_dir"], "ridge": cfg["ridge_dir"]}
    manifests = {"frozen": load_manifest(roots["frozen"], cfg["frozen_manifest_sha256"]),
                 "updated": load_manifest(roots["updated"], cfg["neural_updated_manifest_sha256"]),
                 "tree": load_manifest(roots["tree"], cfg["tree_manifest_sha256"]),
                 "ridge": load_manifest(roots["ridge"], cfg["ridge_manifest_sha256"])}
    coefficient_path = verify(roots["ridge"], manifests["ridge"], "monthly_coefficients.npy")
    coefficients = np.load(coefficient_path)
    factor_names, factor_values = load_zip(cfg["rf_zip"], cfg["evaluation_end"], factor=True)
    rf_index = factor_names.index("RF")
    rf = {month: float(value[rf_index]) for month, value in factor_values.items()}

    records = {(model, arm, weighting, cost): [] for model in cfg["models"] for arm in cfg["arms"]
               for weighting in cfg["portfolio"]["weightings"]
               for cost in cfg["portfolio"]["cost_bps_per_unit_turnover"]}
    previous = {(model, arm, weighting): None for model in cfg["models"] for arm in cfg["arms"]
                for weighting in cfg["portfolio"]["weightings"]}
    coefficient_index = 0
    for year in range(cfg["evaluation_start"] // 100, cfg["evaluation_end"] // 100 + 1):
        arrays = {}
        for name in ["x", "y", "month", "permno"]:
            relative = f"arrays/{year}_{name}.npy"
            path = verify(roots["frozen"], manifests["frozen"], relative)
            arrays[name] = np.load(path)
        evaluation = ((arrays["month"] >= cfg["evaluation_start"]) &
                      (arrays["month"] <= cfg["evaluation_end"]))
        for name in arrays:
            arrays[name] = arrays[name][evaluation]
        frozen_path = verify(roots["frozen"], manifests["frozen"], f"{year}_seed_predictions.npy")
        neural_frozen = np.load(frozen_path).mean(axis=1)
        tree_path = verify(roots["tree"], manifests["tree"], f"{year}_predictions.npz")
        tree = np.load(tree_path)
        for key in ["months", "permno", "y"]:
            if not np.array_equal(tree[key], arrays[{"months":"month","permno":"permno","y":"y"}[key]]):
                raise RuntimeError(f"tree alignment mismatch: {year}:{key}")
        if len(neural_frozen) != len(arrays["y"]):
            raise RuntimeError(f"neural frozen alignment mismatch: {year}")
        for month in np.unique(arrays["month"]):
            mask = arrays["month"] == month
            ids, y, x = arrays["permno"][mask], arrays["y"][mask].astype(float), arrays["x"][mask]
            log_cap = x[:, 0].astype(float) * cap_scale + cap_mean
            total_return = y + rf[int(month)]
            design = np.column_stack([np.ones(len(x)), x.astype(float)])
            ridge_prediction = design @ coefficients[coefficient_index].T
            predictions = {
                "neural": {"frozen": neural_frozen[mask]},
                "ridge": {arm: ridge_prediction[:, index] for index, arm in enumerate(cfg["arms"])},
                "tree": {arm: tree[arm][mask] for arm in cfg["arms"]},
            }
            for arm in ["rolling60", "expanding"]:
                relative = f"{int(month)}/{arm}/seed_predictions.npy"
                updated_path = verify(roots["updated"], manifests["updated"], relative)
                updated = np.load(updated_path).mean(axis=1)
                if len(updated) != len(y):
                    raise RuntimeError(f"neural update alignment mismatch: {month}:{arm}")
                predictions["neural"][arm] = updated
            for model in cfg["models"]:
                for arm in cfg["arms"]:
                    prediction = np.asarray(predictions[model][arm], dtype=float)
                    for weighting in cfg["portfolio"]["weightings"]:
                        weights = target_weights(prediction, log_cap, cfg["portfolio"]["deciles"], weighting)
                        gross_return = float(np.sum(weights * y))
                        turn = turnover(ids, weights, previous[(model, arm, weighting)])
                        for cost in cfg["portfolio"]["cost_bps_per_unit_turnover"]:
                            records[(model, arm, weighting, cost)].append({
                                "month": int(month), "gross_return": gross_return, "turnover": turn,
                                "net_return": gross_return - cost / 10000 * turn})
                        previous[(model, arm, weighting)] = (ids.copy(), weights, total_return.copy())
            coefficient_index += 1
    if coefficient_index != 239:
        raise RuntimeError("evaluation calendar mismatch")

    summary = {model: {arm: {weighting: {str(cost): performance(records[(model, arm, weighting, cost)],
                    cfg["portfolio"]["risk_aversion_gamma"])
                    for cost in cfg["portfolio"]["cost_bps_per_unit_turnover"]}
                    for weighting in cfg["portfolio"]["weightings"]} for arm in cfg["arms"]}
                    for model in cfg["models"]}
    primary_weighting, primary_cost = cfg["portfolio"]["primary_weighting"], cfg["portfolio"]["primary_cost_bps"]
    endpoints, order = {}, []
    for model in cfg["models"]:
        frozen = np.asarray([row["net_return"] for row in records[(model, "frozen", primary_weighting, primary_cost)]])
        for arm in ["rolling60", "expanding"]:
            updated = np.asarray([row["net_return"] for row in records[(model, arm, primary_weighting, primary_cost)]])
            key = f"{model}:{arm}_minus_frozen"
            endpoints[key], order = updated - frozen, order + [key]
    infer_cfg = cfg["inference"]
    inference = {key: block_test(endpoints[key], infer_cfg["bootstrap_replications"], infer_cfg["block_months"],
                                 infer_cfg["seed"] + index) for index, key in enumerate(order)}
    adjusted = multipletests([inference[key]["p_two_sided_centered"] for key in order], method="holm")[1]
    for key, value in zip(order, adjusted):
        inference[key]["p_holm"] = float(value)
    report = {"experiment_id": cfg["experiment_id"], "scope": cfg["scope"],
              "claim_boundary": cfg["claim_boundary"], "months": coefficient_index,
              "portfolio_summary": summary, "primary_net_return_inference": inference,
              "archive_verification": "pass"}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    flat = [{"model": key[0], "arm": key[1], "weighting": key[2], "cost_bps": key[3], **row}
            for key, values in records.items() for row in values]
    (out / "monthly_portfolios.json").write_text(json.dumps(flat, ensure_ascii=False, indent=2) + "\n")
    manifest = {"config_sha256": sha(args.config), "script_sha256": sha(__file__),
                "input_manifests": {"frozen": cfg["frozen_manifest_sha256"],
                    "updated": cfg["neural_updated_manifest_sha256"], "tree": cfg["tree_manifest_sha256"],
                    "ridge": cfg["ridge_manifest_sha256"]},
                "runtime": {"python": platform.python_version(), "executable": sys.executable,
                            "numpy": np.__version__, "statsmodels": statsmodels.__version__},
                "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()}}
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"months": coefficient_index, "primary_inference": inference}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
