"""Machine Learning features, model bundle, and business logic for health indexing."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np

from pdm.core import INFORMATIVE_SENSORS, RAW_COLUMNS, HealthThresholds, PlantEconomics

# --- FEATURE ENGINEERING ---
DEFAULT_WINDOW = 20
_SENSOR_IDX = np.array([RAW_COLUMNS.index(s) for s in INFORMATIVE_SENSORS])


def feature_names(sensors: list[str] = INFORMATIVE_SENSORS) -> list[str]:
    return ["cycle"] + [f"{s}_{stat}" for s in sensors for stat in ("last", "mean", "std", "slope")]


FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)


def window_features(window: np.ndarray, cycle: float) -> np.ndarray:
    if window.ndim != 2 or len(window) == 0:
        raise ValueError("window must be a non-empty 2-D array")

    x = window[:, _SENSOR_IDX].astype(float)
    n = len(x)
    last, mean = x[-1], x.mean(0)
    std = x.std(0) if n > 1 else np.zeros_like(last)

    t = np.arange(n, dtype=float) - (n - 1) / 2
    slope = (t[:, None] * (x - mean)).sum(0) / (t**2).sum() if n > 1 else np.zeros_like(last)

    out = np.empty(1 + 4 * x.shape[1], dtype=float)
    out[0] = cycle
    out[1::4], out[2::4], out[3::4], out[4::4] = last, mean, std, slope
    return out


def trajectory_features(rows: np.ndarray, window: int = DEFAULT_WINDOW) -> np.ndarray:
    return np.array(
        [window_features(rows[max(0, i + 1 - window) : i + 1, 2:], c) for i, c in enumerate(rows[:, 1])]
    )


def dataset_features(
    arr: np.ndarray, window: int = DEFAULT_WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, unit ids, cycles) for every row of a C-MAPSS array."""
    from pdm.simulation import iter_units

    feats, units, cycles = zip(
        *[
            (trajectory_features(rows, window), np.full(len(rows), u), rows[:, 1])
            for u, rows in iter_units(arr)
        ],
        strict=True,
    )
    return np.vstack(feats), np.concatenate(units), np.concatenate(cycles)


def healthy_vector(window: np.ndarray) -> np.ndarray:
    return window_features(window, 0.0)


# --- ML MODELS ---
ARTIFACT_FILES = ("metadata.json", "rul_p50.txt", "interval.json", "anomaly.npz")


@dataclass
class AnomalyDetector:
    mean: np.ndarray
    inv_cov: np.ndarray
    scale: np.ndarray
    threshold: float
    baseline_score: float = 0.75

    @classmethod
    def fit(cls, healthy: np.ndarray, percentile: float = 99.0, ridge: float = 1e-3) -> AnomalyDetector:
        std = healthy.std(0)
        scale = np.where(std == 0, 1.0, std)
        z = (healthy - healthy.mean(0)) / scale
        det = cls(
            healthy.mean(0),
            np.linalg.inv(np.cov(z, rowvar=False) + ridge * np.eye(z.shape[1])),
            scale,
            1.0,
        )
        d = det.distance(healthy)
        det.threshold = float(np.percentile(d, percentile))
        det.baseline_score = float(np.median(d) / det.threshold)
        return det

    def distance(self, x: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(x) - self.mean) / self.scale
        return np.sqrt(np.einsum("ij,jk,ik->i", z, self.inv_cov, z))

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.distance(x) / self.threshold

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            inv_cov=self.inv_cov,
            scale=self.scale,
            threshold=self.threshold,
            baseline_score=self.baseline_score,
        )

    @classmethod
    def load(cls, path: Path) -> AnomalyDetector:
        with np.load(path) as z:
            baseline = float(z["baseline_score"]) if "baseline_score" in z.files else 0.75
            return cls(z["mean"], z["inv_cov"], z["scale"], float(z["threshold"]), baseline)


@dataclass
class ConformalInterval:
    edges: list[float]
    lo: list[float]
    hi: list[float]
    alpha: float = 0.1

    @classmethod
    def fit(
        cls, y_hat: np.ndarray, y_true: np.ndarray, rul_cap: float, n_buckets: int = 6, alpha: float = 0.1
    ) -> ConformalInterval:
        edges = np.linspace(0.0, rul_cap, n_buckets + 1).tolist()
        edges[-1] = float("inf")
        resid, lo, hi = y_true - y_hat, [], []

        for i in range(n_buckets):
            r = resid[(y_hat >= edges[i]) & (y_hat < edges[i + 1])]
            r = r if len(r) >= 30 else resid
            lo.append(min(0.0, float(np.quantile(r, alpha))))
            hi.append(max(0.0, float(np.quantile(r, 1 - alpha))))

        return cls(edges, lo, hi, alpha)

    def bounds(self, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_hat = np.atleast_1d(y_hat)
        idx = np.clip(np.searchsorted(self.edges, y_hat, side="right") - 1, 0, len(self.lo) - 1)
        return y_hat + np.array(self.lo)[idx], y_hat + np.array(self.hi)[idx]

    def to_dict(self) -> dict:
        return {
            "edges": [e if np.isfinite(e) else 1e9 for e in self.edges],
            "lo": self.lo,
            "hi": self.hi,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConformalInterval:
        return cls([float(e) for e in d["edges"]], d["lo"], d["hi"], d.get("alpha", 0.1))


@dataclass
class Prediction:
    rul_p10: float
    rul_p50: float
    rul_p90: float
    anomaly_score: float
    anomaly_ready: bool = True


@dataclass
class ModelBundle:
    booster: lgb.Booster
    interval: ConformalInterval
    anomaly: AnomalyDetector
    metadata: dict = field(default_factory=dict)

    @property
    def window(self) -> int:
        return int(self.metadata.get("window", DEFAULT_WINDOW))

    @property
    def rul_cap(self) -> float:
        return float(self.metadata.get("rul_cap", 125.0))

    @property
    def anomaly_min_history(self) -> int:
        return int(self.metadata.get("anomaly_min_history", max(1, int(self.window * 0.75))))

    def predict_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p50 = np.clip(self.booster.predict(np.atleast_2d(X)), 0, self.rul_cap)
        lo, hi = self.interval.bounds(p50)
        return np.clip(lo, 0, self.rul_cap), p50, np.clip(hi, 0, self.rul_cap)

    def predict_window(self, window: np.ndarray, cycle: float) -> Prediction:
        w = np.asarray(window, dtype=float)[-self.window :]
        lo, mid, hi = (float(v[0]) for v in self.predict_features(window_features(w, cycle)[None, :]))
        ready = len(w) >= self.anomaly_min_history
        score = float(self.anomaly.score(healthy_vector(w)[None, :])[0]) if ready else 0.0
        return Prediction(lo, mid, hi, score, ready)

    def save(self, directory: Path | str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "rul_p50.txt"))
        (d / "interval.json").write_text(json.dumps(self.interval.to_dict(), indent=2))
        self.anomaly.save(d / "anomaly.npz")
        (d / "metadata.json").write_text(
            json.dumps(
                {"feature_names": FEATURE_NAMES, "window": self.window, "rul_cap": self.rul_cap, **self.metadata},
                indent=2,
                default=str,
            )
        )
        return d

    @classmethod
    def load(cls, directory: Path | str) -> ModelBundle:
        d = Path(directory)
        return cls(
            lgb.Booster(model_file=str(d / "rul_p50.txt")),
            ConformalInterval.from_dict(json.loads((d / "interval.json").read_text())),
            AnomalyDetector.load(d / "anomaly.npz"),
            json.loads((d / "metadata.json").read_text()),
        )

    @staticmethod
    def _split_s3_uri(s3_uri: str) -> tuple[str, str]:
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"not an S3 URI: {s3_uri}")
        bucket, _, prefix = s3_uri[5:].partition("/")
        return bucket, prefix.rstrip("/")

    @classmethod
    def load_from_s3(cls, s3_uri: str, cache_dir: Path | str | None = None) -> ModelBundle:
        import boto3

        bucket, prefix = cls._split_s3_uri(s3_uri)
        cache = Path(cache_dir or os.path.join(tempfile.gettempdir(), "pdm-model"))
        cache.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        for n in ARTIFACT_FILES:
            s3.download_file(bucket, f"{prefix}/{n}" if prefix else n, str(cache / n))
        return cls.load(cache)

    # FIX: train.py --upload called this method, but it did not exist.
    def upload_to_s3(self, s3_uri: str) -> None:
        """Save the bundle and upload its artifact files under ``s3_uri`` (s3://bucket/prefix)."""
        import boto3

        bucket, prefix = self._split_s3_uri(s3_uri)
        s3 = boto3.client("s3")
        with tempfile.TemporaryDirectory() as tmp:
            d = self.save(tmp)
            for n in ARTIFACT_FILES:
                s3.upload_file(str(d / n), bucket, f"{prefix}/{n}" if prefix else n)


def load_bundle(model_dir: str | Path, model_s3_uri: str = "") -> ModelBundle:
    return ModelBundle.load_from_s3(model_s3_uri) if model_s3_uri else ModelBundle.load(model_dir)


# --- BUSINESS / HEALTH LOGIC ---
STATUS_ORDER = {"healthy": 0, "warning": 1, "critical": 2}


@dataclass
class HealthAssessment:
    health_index: float
    status: str
    reasons: list[str]
    driver: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


def health_index(rul_p50: float, anomaly_score: float, th: HealthThresholds) -> float:
    life = max(0.0, min(1.0, rul_p50 / th.rul_cap))
    span = max(th.anomaly_critical - th.anomaly_warning, 1e-9)
    penalty = 0.6 * max(0.0, min(1.0, (anomaly_score - th.anomaly_warning) / span))
    return round(100.0 * life * (1.0 - penalty), 1)


def _raw_status(
    rul: float, anom: float, rw: float, rc: float, aw: float, ac: float
) -> tuple[str, list[str], str]:
    reasons, stat, driver = [], "healthy", "none"

    if rul < rc:
        stat, driver = "critical", "rul"
        reasons.append(f"RUL {rul:.0f} cycles < critical threshold {rc:.0f}")
    elif rul < rw:
        stat, driver = "warning", "rul"
        # FIX: this append used to sit outside the elif and ran for every machine.
        reasons.append(f"RUL {rul:.0f} cycles < warning threshold {rw:.0f}")

    if anom >= ac:
        if stat != "critical":
            driver = "anomaly"
        stat = "critical"
        reasons.append(f"Anomaly score {anom:.2f} >= critical {ac:.1f}")
    elif anom >= aw:
        if stat == "healthy":
            stat, driver = "warning", "anomaly"
        reasons.append(f"Anomaly score {anom:.2f} >= warning {aw:.1f}")

    return stat, reasons, driver


def assess(
    rul_p50: float, anomaly_score: float, th: HealthThresholds, previous: str | None = None
) -> HealthAssessment:
    stat, reasons, driver = _raw_status(
        rul_p50, anomaly_score, th.rul_warning, th.rul_critical, th.anomaly_warning, th.anomaly_critical
    )

    if previous in STATUS_ORDER and STATUS_ORDER[stat] < STATUS_ORDER[previous]:
        sticky, s_reasons, s_driver = _raw_status(
            rul_p50,
            anomaly_score,
            th.rul_warning + th.hysteresis_rul,
            th.rul_critical + th.hysteresis_rul,
            th.anomaly_warning - th.hysteresis_anomaly,
            th.anomaly_critical - th.hysteresis_anomaly,
        )
        if STATUS_ORDER[sticky] > STATUS_ORDER[stat]:
            stat = sticky if STATUS_ORDER[sticky] <= STATUS_ORDER[previous] else previous
            reasons = [r + " (holding: inside hysteresis band)" for r in s_reasons]
            driver = s_driver

    return HealthAssessment(health_index(rul_p50, anomaly_score, th), stat, reasons, driver)


def worsened(previous: str | None, current: str) -> bool:
    return current != "healthy" if previous is None else STATUS_ORDER[current] > STATUS_ORDER.get(previous, 0)


def recovered(previous: str | None, current: str) -> bool:
    return previous is not None and STATUS_ORDER[current] < STATUS_ORDER.get(previous, 0)


def maintenance_due(
    now: datetime, rul_p10: float, th: HealthThresholds, econ: PlantEconomics
) -> tuple[datetime, float]:
    cycles = max(0.0, rul_p10 - th.safety_margin_cycles)
    return now + timedelta(hours=cycles * econ.cycle_hours), cycles


def schedule_maintenance(machines: list[dict], now: datetime, econ: PlantEconomics) -> list[dict]:
    def _dt(v):
        return v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))

    crews = [now] * max(1, econ.maintenance_crews)
    dur = timedelta(hours=econ.planned_outage_hours)
    plan = []

    for m in sorted(machines, key=lambda m: _dt(m["maintenance_due"])):
        due = _dt(m["maintenance_due"])
        c_idx = crews.index(min(crews))
        start = max(max(crews[c_idx], now), due - dur)

        end = start + dur
        crews[c_idx] = end
        slack = (due - end).total_seconds() / 3600.0

        prio = (
            "P1"
            if m.get("status") == "critical" or slack < 0
            else "P2"
            if m.get("status") == "warning"
            else "P3"
        )
        plan.append(
            {
                "machine_id": m["machine_id"],
                "status": m.get("status", "healthy"),
                "rul_p10": m.get("rul_p10"),
                "due": due,
                "start": start,
                "end": end,
                "crew": f"Crew {c_idx + 1}",
                "slack_hours": round(slack, 1),
                "priority": prio,
            }
        )
    return sorted(plan, key=lambda r: (r["priority"], r["start"]))


def roi_estimate(
    n_machines: int,
    unplanned_events_per_machine_year: float,
    pdm_capture_rate: float,
    econ: PlantEconomics,
    platform_cost_per_year: float,
) -> dict:
    events = n_machines * unplanned_events_per_machine_year
    captured = events * pdm_capture_rate
    saving = (econ.unplanned_outage_hours - econ.planned_outage_hours) * econ.downtime_cost_per_hour
    gross = captured * saving
    net = gross - platform_cost_per_year
    return {
        "events_per_year": events,
        "captured_events": captured,
        "saving_per_event": saving,
        "gross_saving": gross,
        "platform_cost": platform_cost_per_year,
        "net_benefit": net,
        "roi_pct": (net / platform_cost_per_year * 100.0) if platform_cost_per_year else float("inf"),
        "payback_months": (platform_cost_per_year / gross * 12.0) if gross > 0 else float("inf"),
        "downtime_hours_avoided": captured * (econ.unplanned_outage_hours - econ.planned_outage_hours),
    }