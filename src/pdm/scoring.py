"""Stream scoring: telemetry messages -> RUL / anomaly / health -> store, archive and alerts.

NOTE: reconstructed. ``pdm/scoring.py`` was missing from the codebase dump. If you still have the
original, keep it and only use this file as a reference for the interface that
``services/scorer/handler.py`` expects:

    StreamScorer(model, store, settings, archive_sink=None, notify_sink=None).process(messages) -> dict
    firehose_sink(stream_name, region) -> callable(list[dict])
    sns_sink(topic_arn, region)        -> callable(list[dict])
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from pdm.core import Settings, TelemetryMessage
from pdm.engine import ModelBundle, assess, maintenance_due, recovered, worsened
from pdm.storage import TelemetryStore, raw_vector_from_item

log = logging.getLogger("pdm.scoring")

Sink = Callable[[list[dict]], None]


class StreamScorer:
    """Keeps a rolling window per machine, scores every message and persists the results."""

    def __init__(
        self,
        model: ModelBundle,
        store: TelemetryStore,
        settings: Settings,
        archive_sink: Sink | None = None,
        notify_sink: Sink | None = None,
    ):
        self.model = model
        self.store = store
        self.settings = settings
        self.archive_sink = archive_sink
        self.notify_sink = notify_sink
        self._buffers: dict[str, deque[np.ndarray]] = {}
        self._last_cycle: dict[str, int] = {}
        self._ewma: dict[str, float | None] = {}
        self._status: dict[str, str | None] = {}

    # -- per-machine cache ---------------------------------------------------------------
    def _warm_start(self, machine_id: str) -> None:
        """Rebuild window + state from the store (new Lambda container / first sight of a machine)."""
        buf: deque[np.ndarray] = deque(maxlen=self.model.window)
        last_cycle, status, ewma = 0, None, None
        try:
            for item in self.store.get_telemetry(machine_id, limit=self.model.window):
                buf.append(np.asarray(raw_vector_from_item(item), dtype=float))
                last_cycle = int(item.get("cycle", last_cycle))
            state = self.store.get_machine_state(machine_id)
            if state:
                status = state.get("status")
                ewma = state.get("anomaly_ewma")
        except Exception:
            log.exception("warm start failed for %s; starting cold", machine_id)
        self._buffers[machine_id] = buf
        self._last_cycle[machine_id] = last_cycle
        self._status[machine_id] = status
        self._ewma[machine_id] = None if ewma is None else float(ewma)

    def _reset(self, machine_id: str) -> None:
        self._buffers[machine_id] = deque(maxlen=self.model.window)
        self._last_cycle[machine_id] = 0
        self._status[machine_id] = None
        self._ewma[machine_id] = None

    def _drop(self, machine_id: str) -> None:
        for cache in (self._buffers, self._last_cycle, self._ewma, self._status):
            cache.pop(machine_id, None)

    # -- scoring -------------------------------------------------------------------------
    def process(self, messages: list[TelemetryMessage]) -> dict:
        th, econ = self.settings.thresholds, self.settings.economics
        items: list[dict] = []
        states: dict[str, dict] = {}
        alerts: list[dict] = []
        duplicates = 0
        touched: set[str] = set()

        for msg in messages:
            mid = msg.machine_id
            if mid not in self._buffers:
                self._warm_start(mid)
            touched.add(mid)

            last = self._last_cycle[mid]
            if msg.cycle == 1 and last > 1:
                self._reset(mid)  # machine was overhauled / replaced
            elif msg.cycle <= last:
                duplicates += 1  # redelivered record
                continue

            self._buffers[mid].append(np.asarray(msg.raw_vector(), dtype=float))
            self._last_cycle[mid] = msg.cycle

            pred = self.model.predict_window(np.stack(self._buffers[mid]), msg.cycle)
            prev_ewma = self._ewma[mid]
            if pred.anomaly_ready:
                a = th.anomaly_ewma_alpha
                ewma = pred.anomaly_score if prev_ewma is None else a * pred.anomaly_score + (1 - a) * prev_ewma
            else:
                ewma = prev_ewma if prev_ewma is not None else 0.0
            self._ewma[mid] = ewma

            previous = self._status[mid]
            health = assess(pred.rul_p50, ewma, th, previous)
            self._status[mid] = health.status

            ts_ms = msg.ts_ms
            due, _ = maintenance_due(datetime.fromtimestamp(ts_ms / 1000, UTC), pred.rul_p10, th, econ)

            items.append(
                {
                    "machine_id": mid,
                    "site": msg.site,
                    "line": msg.line,
                    "ts": ts_ms,
                    "cycle": msg.cycle,
                    "settings": dict(msg.settings),
                    "sensors": dict(msg.sensors),
                    "rul_p10": round(pred.rul_p10, 2),
                    "rul_p50": round(pred.rul_p50, 2),
                    "rul_p90": round(pred.rul_p90, 2),
                    "anomaly_score": round(ewma, 4),
                    "health_index": health.health_index,
                    "status": health.status,
                }
            )
            states[mid] = {
                "machine_id": mid,
                "site": msg.site,
                "line": msg.line,
                "status": health.status,
                "health_index": health.health_index,
                "rul_p10": round(pred.rul_p10, 2),
                "rul_p50": round(pred.rul_p50, 2),
                "rul_p90": round(pred.rul_p90, 2),
                "anomaly_score": round(ewma, 4),
                "anomaly_ewma": round(ewma, 6),
                "cycle": msg.cycle,
                "maintenance_due": due.isoformat(),
                "last_ts": ts_ms,
                "reasons": health.reasons,
                "driver": health.driver,
            }

            if worsened(previous, health.status):
                alerts.append(
                    self._alert(msg, "critical" if health.status == "critical" else "warning",
                                "status_change", health.reasons or [f"status {previous} -> {health.status}"],
                                pred.rul_p50, ewma)
                )
            elif recovered(previous, health.status):
                alerts.append(
                    self._alert(msg, "info", "recovery", [f"recovered {previous} -> {health.status}"],
                                pred.rul_p50, ewma)
                )

        try:
            if items:
                self.store.put_telemetry_batch(items)
            for state in states.values():
                self.store.upsert_machine_state(state)
            for alert in alerts:
                self.store.put_alert(alert)
        except Exception:
            # Cached windows/state may be ahead of the store; rebuild them from the store on retry.
            for mid in touched:
                self._drop(mid)
            raise

        # Side channels must never fail the batch (Lambda would retry the whole Kinesis batch).
        if items and self.archive_sink:
            self._safe(self.archive_sink, items, "archive")
        if alerts and self.notify_sink:
            self._safe(self.notify_sink, alerts, "notify")

        return {"messages": len(items), "machines": len(states), "alerts": len(alerts), "duplicates": duplicates}

    @staticmethod
    def _alert(msg: TelemetryMessage, severity: str, kind: str, reasons: list[str],
               rul_p50: float, anomaly: float) -> dict:
        return {
            "machine_id": msg.machine_id,
            "ts": msg.ts_ms,
            "severity": severity,
            "type": kind,
            "status": "open",
            "message": "; ".join(reasons),
            "cycle": msg.cycle,
            "rul_p50": round(rul_p50, 2),
            "anomaly_score": round(anomaly, 4),
            "site": msg.site,
            "line": msg.line,
        }

    @staticmethod
    def _safe(sink: Sink, payload: list[dict], name: str) -> None:
        try:
            sink(payload)
        except Exception:
            log.exception("%s sink failed (ignored)", name)


# --- AWS sinks ---------------------------------------------------------------------------
def firehose_sink(stream_name: str, region: str | None = None) -> Sink:
    import boto3

    client = boto3.client("firehose", region_name=region)

    def sink(items: list[dict]) -> None:
        for i in range(0, len(items), 500):  # PutRecordBatch limit
            chunk = items[i : i + 500]
            resp = client.put_record_batch(
                DeliveryStreamName=stream_name,
                Records=[{"Data": (json.dumps(it, default=str) + "\n").encode()} for it in chunk],
            )
            if resp.get("FailedPutCount"):
                log.warning("firehose: %s of %d records failed", resp["FailedPutCount"], len(chunk))

    return sink


def sns_sink(topic_arn: str, region: str | None = None) -> Sink:
    import boto3

    client = boto3.client("sns", region_name=region)

    def sink(alerts: list[dict]) -> None:
        for a in alerts:
            subject = f"[{a['severity'].upper()}] {a['machine_id']}: {a['type']}"[:100]
            client.publish(TopicArn=topic_arn, Subject=subject, Message=json.dumps(a, default=str))

    return sink