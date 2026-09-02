from __future__ import annotations

import json
import sqlite3
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from pdm.core import SENSOR_COLUMNS, SETTING_COLUMNS, Settings

TelemetryItem = dict[str, Any]
MachineState = dict[str, Any]
Alert = dict[str, Any]


def raw_vector_from_item(item: TelemetryItem) -> list[float]:
    s, v = item.get("settings", {}), item.get("sensors", {})
    return [float(s.get(c, 0.0)) for c in SETTING_COLUMNS] + [float(v.get(c, 0.0)) for c in SENSOR_COLUMNS]


class TelemetryStore(Protocol):
    def put_telemetry_batch(self, items: list[TelemetryItem]) -> None: ...
    def get_telemetry(self, machine_id: str, limit: int = 300) -> list[TelemetryItem]: ...
    def upsert_machine_state(self, state: MachineState) -> None: ...
    def get_machine_state(self, machine_id: str) -> MachineState | None: ...
    def list_machine_states(self) -> list[MachineState]: ...
    def put_alert(self, alert: Alert) -> None: ...
    def list_alerts(self, status: str | None = None, limit: int = 200) -> list[Alert]: ...
    def ack_alert(self, machine_id: str, ts: int, user: str = "operator") -> None: ...


# --- DYNAMODB BACKEND ---
def _to_ddb(obj: Any) -> Any:
    if isinstance(obj, float): return Decimal(str(obj))
    if isinstance(obj, dict): return {k: _to_ddb(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_ddb(v) for v in obj]
    return obj

def _from_ddb(obj: Any) -> Any:
    if isinstance(obj, Decimal): return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict): return {k: _from_ddb(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_from_ddb(v) for v in obj]
    return obj


class DynamoStore:
    def __init__(self, telemetry_table: str, machine_state_table: str, alerts_table: str, region: str | None = None, ttl_days: int = 7):
        import boto3
        ddb = boto3.resource("dynamodb", region_name=region)
        self.telemetry, self.state, self.alerts, self.ttl_days = ddb.Table(telemetry_table), ddb.Table(machine_state_table), ddb.Table(alerts_table), ttl_days
        from boto3.dynamodb.conditions import Key
        self.Key = Key

    def put_telemetry_batch(self, items: list[TelemetryItem]) -> None:
        ttl = int(time.time()) + self.ttl_days * 86400
        with self.telemetry.batch_writer(overwrite_by_pkeys=["machine_id", "ts"]) as bw:
            for it in items: bw.put_item(Item=_to_ddb({**it, "ttl": ttl}))

    def get_telemetry(self, machine_id: str, limit: int = 300) -> list[TelemetryItem]:
        resp = self.telemetry.query(KeyConditionExpression=self.Key("machine_id").eq(machine_id), ScanIndexForward=False, Limit=limit)
        return [_from_ddb(i) for i in reversed(resp.get("Items", []))]

    def upsert_machine_state(self, state: MachineState) -> None:
        self.state.put_item(Item=_to_ddb(state))

    def get_machine_state(self, machine_id: str) -> MachineState | None:
        return _from_ddb(self.state.get_item(Key={"machine_id": machine_id}).get("Item"))

    def list_machine_states(self) -> list[MachineState]:
        items, kwargs = [], {}
        while True:
            resp = self.state.scan(**kwargs)
            items.extend(resp.get("Items", []))
            if "LastEvaluatedKey" not in resp: break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return sorted((_from_ddb(i) for i in items), key=lambda s: s.get("machine_id", ""))

    def put_alert(self, alert: Alert) -> None:
        self.alerts.put_item(Item=_to_ddb(alert))

    def list_alerts(self, status: str | None = None, limit: int = 200) -> list[Alert]:
        if status:
            resp = self.alerts.query(IndexName="status-ts-index", KeyConditionExpression=self.Key("status").eq(status), ScanIndexForward=False, Limit=limit)
            return [_from_ddb(i) for i in resp.get("Items", [])]
        items, kwargs = [], {}
        while len(items) < limit:
            resp = self.alerts.scan(**kwargs)
            items.extend(resp.get("Items", []))
            if "LastEvaluatedKey" not in resp: break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return [_from_ddb(i) for i in sorted(items, key=lambda a: a.get("ts", 0), reverse=True)[:limit]]

    def ack_alert(self, machine_id: str, ts: int, user: str = "operator") -> None:
        self.alerts.update_item(
            Key={"machine_id": machine_id, "ts": int(ts)},
            UpdateExpression="SET #s = :s, acknowledged_by = :u, acknowledged_at = :t",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": "acknowledged", ":u": user, ":t": int(time.time() * 1000)},
        )


# --- SQLITE BACKEND ---
class SQLiteStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS telemetry (machine_id TEXT NOT NULL, ts INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY (machine_id, ts));
    CREATE TABLE IF NOT EXISTS machine_state (machine_id TEXT PRIMARY KEY, body TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS alerts (machine_id TEXT NOT NULL, ts INTEGER NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY (machine_id, ts));
    CREATE INDEX IF NOT EXISTS alerts_status_ts ON alerts(status, ts);
    """

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:": Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def put_telemetry_batch(self, items: list[TelemetryItem]) -> None:
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO telemetry(machine_id, ts, body) VALUES (?, ?, ?)", [(i["machine_id"], int(i["ts"]), json.dumps(i)) for i in items])
            self._conn.commit()

    def get_telemetry(self, machine_id: str, limit: int = 300) -> list[TelemetryItem]:
        with self._lock:
            rows = self._conn.execute("SELECT body FROM telemetry WHERE machine_id=? ORDER BY ts DESC LIMIT ?", (machine_id, limit)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def upsert_machine_state(self, state: MachineState) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO machine_state(machine_id, body) VALUES (?, ?)", (state["machine_id"], json.dumps(state)))
            self._conn.commit()

    def get_machine_state(self, machine_id: str) -> MachineState | None:
        with self._lock:
            row = self._conn.execute("SELECT body FROM machine_state WHERE machine_id=?", (machine_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_machine_states(self) -> list[MachineState]:
        with self._lock:
            return [json.loads(r[0]) for r in self._conn.execute("SELECT body FROM machine_state ORDER BY machine_id").fetchall()]

    def put_alert(self, alert: Alert) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO alerts(machine_id, ts, status, body) VALUES (?, ?, ?, ?)", (alert["machine_id"], int(alert["ts"]), alert.get("status", "open"), json.dumps(alert)))
            self._conn.commit()

    def list_alerts(self, status: str | None = None, limit: int = 200) -> list[Alert]:
        with self._lock:
            query = "SELECT body FROM alerts WHERE status=? ORDER BY ts DESC LIMIT ?" if status else "SELECT body FROM alerts ORDER BY ts DESC LIMIT ?"
            args = (status, limit) if status else (limit,)
            return [json.loads(r[0]) for r in self._conn.execute(query, args).fetchall()]

    def ack_alert(self, machine_id: str, ts: int, user: str = "operator") -> None:
        with self._lock:
            if row := self._conn.execute("SELECT body FROM alerts WHERE machine_id=? AND ts=?", (machine_id, int(ts))).fetchone():
                body = json.loads(row[0]) | {"status": "acknowledged", "acknowledged_by": user, "acknowledged_at": int(time.time() * 1000)}
                self._conn.execute("UPDATE alerts SET status=?, body=? WHERE machine_id=? AND ts=?", ("acknowledged", json.dumps(body), machine_id, int(ts)))
                self._conn.commit()


def get_store(settings: Settings | None = None) -> TelemetryStore:
    settings = settings or Settings()
    if settings.backend == "dynamodb":
        return DynamoStore(settings.telemetry_table, settings.machine_state_table, settings.alerts_table, settings.aws_region)
    return SQLiteStore(settings.local_db_path)