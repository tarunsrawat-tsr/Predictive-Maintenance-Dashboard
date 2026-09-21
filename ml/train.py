"""Train the RUL model, its conformal intervals, and the anomaly detector on NASA C-MAPSS."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm.engine import (  # noqa: E402
    FEATURE_NAMES,
    AnomalyDetector,
    ConformalInterval,
    ModelBundle,
    dataset_features,
    healthy_vector,
    window_features,
)
from pdm.simulation import add_rul_labels, download, iter_units, load_rul, load_split  # noqa: E402

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}
RUL_CAP = 125.0
HEALTHY_FRAC = 0.35


def nasa_score(d: np.ndarray) -> float:
    """NASA PHM08 asymmetric score; d = predicted - true."""
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FD001")
    ap.add_argument("--out", default="artifacts/model")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--data-dir", default="data/cmapss")
    ap.add_argument("--upload", default="")
    args = ap.parse_args()
    t0 = time.perf_counter()

    download(args.data_dir)
    train = load_split("train", args.dataset, args.data_dir)
    test = load_split("test", args.dataset, args.data_dir)

    # Features & labels (same unit ordering for both)
    X, units, _ = dataset_features(train, args.window)
    y = add_rul_labels(np.vstack([r for _, r in iter_units(train)]), cap=RUL_CAP)
    assert len(X) == len(y), "feature/label row mismatch"

    # Hold-out validation (by engine unit)
    uniq = np.unique(units)
    val_units = np.random.default_rng(0).choice(uniq, size=int(0.25 * len(uniq)), replace=False)
    val = np.isin(units, val_units)

    holdout = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X[~val], y[~val]),
        num_boost_round=800,
        valid_sets=[lgb.Dataset(X[val], y[val])],
        callbacks=[lgb.early_stopping(60, False)],
    )

    # Calibrate intervals & refit final model
    y_va_hat = np.clip(holdout.predict(X[val]), 0, RUL_CAP)
    interval = ConformalInterval.fit(y_va_hat, y[val], RUL_CAP)
    lo_v, hi_v = interval.bounds(y_va_hat)
    best_it = int(holdout.best_iteration or 800)
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(X, y), num_boost_round=best_it)

    # Anomaly baseline (early life of engines)
    healthy = np.array(
        [
            healthy_vector(rows[i + 1 - args.window : i + 1, 2:])
            for _, rows in iter_units(train)
            for i in range(args.window - 1, max(args.window, int(len(rows) * HEALTHY_FRAC)))
        ]
    )
    anomaly = AnomalyDetector.fit(healthy)

    bundle = ModelBundle(
        booster=booster,
        interval=interval,
        anomaly=anomaly,
        metadata={"window": args.window, "rul_cap": RUL_CAP, "dataset": args.dataset},
    )

    # Test evaluation (last available window of each test engine)
    test_units = [rows for _, rows in iter_units(test)]
    Xt_last = np.array([window_features(rows[-args.window :, 2:], rows[-1, 1]) for rows in test_units])
    y_raw = load_rul(args.dataset, args.data_dir)
    y_true = np.minimum(y_raw, RUL_CAP)
    t_lo, t_hat, t_hi = bundle.predict_features(Xt_last)
    d = t_hat - y_true
    end_scores = anomaly.score(np.array([healthy_vector(rows[-args.window :, 2:]) for rows in test_units]))

    gain = booster.feature_importance("gain")
    top = np.argsort(gain)[::-1][:15]
    nasa = nasa_score(d)

    metrics = {
        "dataset": args.dataset,
        "window": args.window,
        "rul_cap": RUL_CAP,
        "n_train_rows": int(len(X)),
        "n_train_units": int(len(uniq)),
        "n_calibration_units": int(len(val_units)),
        "best_iteration": best_it,
        "holdout_rmse": float(np.sqrt(np.mean((y_va_hat - y[val]) ** 2))),
        "holdout_coverage_p10_p90": float(np.mean((np.clip(lo_v, 0, RUL_CAP) <= y[val]) & (y[val] <= np.clip(hi_v, 0, RUL_CAP)))),
        "test_rmse_capped": float(np.sqrt(np.mean(d**2))),
        "test_rmse_uncapped": float(np.sqrt(np.mean((t_hat - y_raw) ** 2))),
        "test_nasa_score_capped": nasa,
        "test_nasa_score": nasa,
        "test_coverage_p10_p90": float(np.mean((t_lo <= y_true) & (y_true <= t_hi))),
        "test_mean_interval_width": float(np.mean(t_hi - t_lo)),
        "anomaly_score_early_life_mean": float(np.mean(anomaly.score(healthy))),
        "anomaly_score_end_of_trace_mean": float(np.mean(end_scores)),
        "conformal_offsets": interval.to_dict(),
        "top_features_gain": [{"feature": FEATURE_NAMES[i], "gain": float(gain[i])} for i in top],
        "train_seconds": round(time.perf_counter() - t0, 1),
    }

    bundle.metadata.update({"trained_at": datetime.now(UTC).isoformat(), "metrics": metrics})
    out = bundle.save(args.out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {out}\n{json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}, indent=2)}")

    if args.upload:
        bundle.upload_to_s3(args.upload)


if __name__ == "__main__":
    main()