"""Annual shallow boosted-tree frozen/rolling/expanding update comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_gib():
    """Normalize getrusage's platform-specific maximum-RSS unit to GiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
    return value / divisor


def month_weights(months):
    _, inverse, counts = np.unique(months, return_inverse=True, return_counts=True)
    value = 1.0 / counts[inverse]
    return (value / value.mean()).astype(np.float64)


def training_mask(months, first_target, arm, rolling_months):
    ordinal = (months // 100) * 12 + months % 100
    target_ordinal = (first_target // 100) * 12 + first_target % 100
    historical = months < first_target
    if arm == "expanding":
        mask = historical
    elif arm == "rolling60":
        mask = historical & (ordinal >= target_ordinal - rolling_months)
        if len(np.unique(months[mask])) != rolling_months:
            raise RuntimeError("incomplete rolling calendar")
    else:
        raise ValueError(arm)
    if not mask.any() or np.max(months[mask]) >= first_target:
        raise RuntimeError("nonhistorical training slice")
    return mask


def fit_model(x, y, months, cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = dict(cfg["hyperparameters"])
    model = HistGradientBoostingRegressor(**params)
    model.fit(x, y, sample_weight=month_weights(months))
    return model


def components(y, frozen, updated):
    y, frozen, updated = [np.asarray(value, dtype=np.float64) for value in (y, frozen, updated)]
    error = y - frozen
    increment = updated - frozen
    a = float(np.mean(increment ** 2))
    b = float(np.mean(error * increment))
    direct = float(np.mean((y - updated) ** 2 - (y - frozen) ** 2))
    return {"adjustment_cost_A": a, "alignment_B": b, "alignment_benefit_2B": 2 * b,
            "updated_minus_frozen_loss": direct, "identity_error": direct - (a - 2 * b)}


def summarize(rows, arm):
    loss = {name: float(np.mean([row["loss"][name] for row in rows])) for name in ["zero", "frozen", arm]}
    a = float(np.mean([row[arm]["adjustment_cost_A"] for row in rows]))
    b = float(np.mean([row[arm]["alignment_B"] for row in rows]))
    return {"months": len(rows), "stock_months": int(sum(row["stocks"] for row in rows)), "loss": loss,
            "r2_against_zero": {name: 1 - loss[name] / loss["zero"] for name in ["frozen", arm]},
            "adjustment_cost_A": a, "alignment_benefit_2B": 2 * b,
            "updated_minus_frozen_loss": loss[arm] - loss["frozen"],
            "ex_post_lambda_star": float(np.clip(b / a, 0, 1)) if a > 0 else 0.0}


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        if int(os.environ[variable]) != cfg["threads"]:
            raise RuntimeError(f"thread setting mismatch: {variable}")
    source = Path(cfg["source_dir"])
    manifest_path = source / "output_manifest.json"
    if sha(manifest_path) != cfg["source_manifest_sha256"]:
        raise RuntimeError("source manifest mismatch")
    parent_outputs = json.loads(manifest_path.read_text())["outputs"]
    years = sorted(int(path.name.split("_")[0]) for path in (source / "arrays").glob("*_month.npy"))
    final_year = cfg["evaluation_end"] // 100
    years = [year for year in years if year <= final_year]
    arrays = {name: [] for name in ["x", "y", "month", "permno"]}
    verified = {}
    started = time.time()
    for year in years:
        for name in arrays:
            path = source / "arrays" / f"{year}_{name}.npy"
            relative = str(path.relative_to(source))
            if relative not in parent_outputs or sha(path) != parent_outputs[relative]:
                raise RuntimeError(f"array checksum mismatch: {relative}")
            verified[relative] = parent_outputs[relative]
            arrays[name].append(np.load(path))
    arrays = {name: np.concatenate(values) for name, values in arrays.items()}
    order = np.argsort(arrays["month"], kind="stable")
    arrays = {name: value[order] for name, value in arrays.items()}
    if not np.isfinite(arrays["x"]).all() or not np.isfinite(arrays["y"]).all():
        raise RuntimeError("nonfinite source array")
    base = arrays["month"] <= cfg["base_training_end"]
    if not base.any() or np.max(arrays["month"][base]) != cfg["base_training_end"]:
        raise RuntimeError("base training endpoint mismatch")
    model_started = time.time()
    frozen_model = fit_model(arrays["x"][base], arrays["y"][base], arrays["month"][base], cfg)
    fit_records = [{"arm": "frozen", "target_year": None, "rows": int(base.sum()),
                    "first_month": int(arrays["month"][base].min()), "last_month": int(arrays["month"][base].max()),
                    "seconds": time.time() - model_started}]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for year in range(cfg["evaluation_start"] // 100, final_year + 1):
        first_target = max(cfg["evaluation_start"], year * 100 + 1)
        targets = ((arrays["month"] >= first_target) &
                   (arrays["month"] <= min(cfg["evaluation_end"], year * 100 + 12)))
        if not targets.any():
            continue
        models = {"frozen": frozen_model}
        for arm in ["rolling60", "expanding"]:
            mask = training_mask(arrays["month"], first_target, arm, cfg["rolling_months"])
            if arm == "expanding" and np.array_equal(mask, base):
                models[arm] = frozen_model
                seconds = 0.0
            else:
                fit_started = time.time()
                models[arm] = fit_model(arrays["x"][mask], arrays["y"][mask], arrays["month"][mask], cfg)
                seconds = time.time() - fit_started
            fit_records.append({"arm": arm, "target_year": year, "rows": int(mask.sum()),
                                "first_month": int(arrays["month"][mask].min()),
                                "last_month": int(arrays["month"][mask].max()), "seconds": seconds})
        x_eval, y_eval = arrays["x"][targets], arrays["y"][targets]
        prediction = {arm: model.predict(x_eval).astype(np.float32) for arm, model in models.items()}
        eval_months, eval_permno = arrays["month"][targets], arrays["permno"][targets]
        np.savez_compressed(out / f"{year}_predictions.npz", months=eval_months, permno=eval_permno,
                            y=y_eval.astype(np.float32), **prediction)
        for month in np.unique(eval_months):
            mask = eval_months == month
            frozen = prediction["frozen"][mask]
            y = y_eval[mask]
            row = {"month": int(month), "stocks": int(mask.sum()),
                   "loss": {"zero": float(np.mean(y.astype(np.float64) ** 2)),
                            "frozen": float(np.mean((y - frozen).astype(np.float64) ** 2))}}
            for arm in ["rolling60", "expanding"]:
                updated = prediction[arm][mask]
                row["loss"][arm] = float(np.mean((y - updated).astype(np.float64) ** 2))
                row[arm] = components(y, frozen, updated)
                if abs(row[arm]["identity_error"]) > 1e-12:
                    raise RuntimeError(f"loss identity failed: {month}:{arm}")
            rows.append(row)
        print(f"completed tree target year {year}", flush=True)
    expected = [year * 100 + month for year in range(cfg["evaluation_start"] // 100, final_year + 1)
                for month in range(1, 13)
                if cfg["evaluation_start"] <= year * 100 + month <= cfg["evaluation_end"]]
    if [row["month"] for row in rows] != expected:
        raise RuntimeError("evaluation calendar mismatch")
    import sklearn
    report = {
        "experiment_id": cfg["experiment_id"], "scope": cfg["scope"],
        "months": len(rows), "stock_months": int(sum(row["stocks"] for row in rows)),
        "summary": {arm: summarize(rows, arm) for arm in ["rolling60", "expanding"]},
        "fits": fit_records, "elapsed_seconds": time.time() - started,
        "peak_rss_gib": peak_rss_gib(),
    }
    (out / "monthly_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    final = {"config_sha256": sha(args.config), "script_sha256": sha(__file__),
             "parent_manifest_sha256": cfg["source_manifest_sha256"], "verified_inputs": verified,
             "runtime": {"python": platform.python_version(), "executable": sys.executable,
                         "numpy": np.__version__, "sklearn": sklearn.__version__,
                         "threads": cfg["threads"]},
             "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()}}
    (out / "output_manifest.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
