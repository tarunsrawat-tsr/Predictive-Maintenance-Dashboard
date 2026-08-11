from __future__ import annotations

import io
import logging
import os
import random
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pdm.core import SENSOR_COLUMNS, SETTING_COLUMNS, TelemetryMessage, utc_now_iso

log = logging.getLogger("pdm.simulation")

DEFAULT_SITE = "nagoya"
DEFAULT_LINES = ["L1", "L2", "L3"]

# NASA Prognostics Data Repository (override with PDM_CMAPSS_URL if the link moves).
CMAPSS_URL = os.environ.get(
    "PDM_CMAPSS_URL",
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
)
N_COLUMNS = 2 + len(SETTING_COLUMNS) + len(SENSOR_COLUMNS)  # unit, cycle, 3 settings, 21 sensors = 26


# --- DATA LOADING (restored: these were missing after the refactor) ---
def _extract(fileobj, dest: Path, depth: int = 0) -> None:
    """Extract every .txt from a (possibly nested) zip, flattened into ``dest``."""
    with zipfile.ZipFile(fileobj) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name  # flatten; also prevents zip-slip
            data = zf.read(info)
            if name.lower().endswith(".zip") and depth < 2:
                _extract(io.BytesIO(data), dest, depth + 1)
            elif name.lower().endswith(".txt"):
                (dest / name).write_bytes(data)


def download(data_dir: str | Path = "data/cmapss") -> Path:
    """Download and unpack C-MAPSS into ``data_dir`` (no-op if FD001 is already there)."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "train_FD001.txt").exists():
        return d
    log.info("downloading C-MAPSS from %s", CMAPSS_URL)
    with urllib.request.urlopen(CMAPSS_URL, timeout=120) as resp:  # noqa: S310
        blob = resp.read()
    _extract(io.BytesIO(blob), d)
    if not (d / "train_FD001.txt").exists():
        raise RuntimeError(f"C-MAPSS archive did not contain train_FD001.txt (see {CMAPSS_URL})")
    return d


def load_split(split: str, dataset: str = "FD001", data_dir: str | Path = "data/cmapss") -> np.ndarray:
    """Load ``train`` or ``test`` rows as an (n, 26) array: unit, cycle, 3 settings, 21 sensors."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = Path(data_dir) / f"{split}_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - call pdm.simulation.download() first")
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != N_COLUMNS:
        raise ValueError(f"{path}: expected {N_COLUMNS} columns, got shape {arr.shape}")
    return arr


def load_rul(dataset: str = "FD001", data_dir: str | Path = "data/cmapss") -> np.ndarray:
    """True RUL at the last observed cycle of each test engine (ordered by unit id)."""
    path = Path(data_dir) / f"RUL_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - call pdm.simulation.download() first")
    return np.loadtxt(path, dtype=float).ravel()


def iter_units(arr: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (unit_id, rows sorted by cycle) in ascending unit order."""
    units = arr[:, 0].astype(int)
    for u in np.unique(units):
        rows = arr[units == u]
        yield int(u), rows[np.argsort(rows[:, 1], kind="stable")]


def add_rul_labels(arr: np.ndarray, cap: float | None = None) -> np.ndarray:
    """Per-row RUL = (last cycle of that unit) - cycle, optionally capped (piecewise-linear RUL)."""
    units = arr[:, 0].astype(int)
    cycles = arr[:, 1]
    _, inv = np.unique(units, return_inverse=True)
    last = np.zeros(inv.max() + 1)
    np.maximum.at(last, inv, cycles)
    rul = last[inv] - cycles
    return np.minimum(rul, cap) if cap is not None else rul


# --- FLEET SIMULATOR ---
@dataclass
class MachineReplay:
    machine_id: str
    site: str
    line: str
    unit: int
    rows: np.ndarray  # (n, 26): unit, cycle, raw...
    pos: int = 0
    overhauls: int = 0

    @property
    def remaining(self) -> int:
        return len(self.rows) - self.pos


@dataclass
class FleetSimulator:
    n_machines: int = 10
    dataset: str = "FD001"
    split: str = "train"
    site: str = DEFAULT_SITE
    lines: list[str] = field(default_factory=lambda: list(DEFAULT_LINES))
    seed: int = 42
    noise_std: float = 0.0
    start_offset: str = "random"  # "random" | "zero"
    machine_prefix: str = "GT"
    data_dir: str | None = None
    virtual_tick_seconds: float = 0.0
    virtual_start: float | None = None

    machines: dict[str, MachineReplay] = field(default_factory=dict, init=False)
    tick: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

        arr = (
            load_split(self.split, self.dataset, self.data_dir)
            if self.data_dir
            else load_split(self.split, self.dataset)
        )
        self._units = dict(iter_units(arr))
        self._pool = list(self._units)
        self._rng.shuffle(self._pool)
        self._sensor_std = arr[:, 2:].std(axis=0)

        if self.virtual_start is None:
            self.virtual_start = time.time()

        for i in range(self.n_machines):
            mid = f"{self.machine_prefix}-{i + 1:03d}"
            line = self.lines[i % len(self.lines)]
            self.machines[mid] = self._new_replay(mid, line)

    def _draw_unit(self) -> int:
        if not self._pool:
            self._pool = list(self._units)
            self._rng.shuffle(self._pool)
        return self._pool.pop()

    def _new_replay(self, machine_id: str, line: str, overhauls: int = 0) -> MachineReplay:
        unit = self._draw_unit()
        rows = self._units[unit]
        # Start somewhere in the first 80% of life so the fleet is staggered.
        pos = self._rng.randint(0, int(len(rows) * 0.8)) if self.start_offset == "random" else 0
        return MachineReplay(machine_id, self.site, line, unit, rows, pos, overhauls)

    def _now_iso(self) -> str:
        if self.virtual_tick_seconds > 0:
            t = (self.virtual_start or 0.0) + self.tick * self.virtual_tick_seconds
            return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return utc_now_iso()

    def _to_message(self, m: MachineReplay) -> TelemetryMessage:
        row = m.rows[m.pos]
        raw = row[2:].astype(float).copy()

        if self.noise_std > 0:
            rng = np.random.default_rng(self.tick * 1000 + m.pos)
            raw += rng.normal(0, self.noise_std, len(raw)) * self._sensor_std

        return TelemetryMessage(
            machine_id=m.machine_id,
            site=m.site,
            line=m.line,
            ts=self._now_iso(),
            cycle=int(row[1]) - int(m.rows[0, 1]) + 1,
            settings=dict(zip(SETTING_COLUMNS, raw[:3], strict=True)),
            sensors=dict(zip(SENSOR_COLUMNS, raw[3:], strict=True)),
        )

    def step(self) -> list[TelemetryMessage]:
        """Advance every machine by one cycle and return the messages to publish."""
        out = []
        for mid, m in self.machines.items():
            if m.pos >= len(m.rows):
                # Failure reached -> overhaul with a fresh unit, restart at cycle 1.
                self.machines[mid] = m = self._new_replay(m.machine_id, m.line, m.overhauls + 1)
                m.pos = 0

            out.append(self._to_message(m))
            m.pos += 1

        self.tick += 1
        return out

    def true_rul(self, machine_id: str) -> int | None:
        """Ground-truth RUL of the current unit (for evaluation/demo overlays only)."""
        if m := self.machines.get(machine_id):
            return max(0, len(m.rows) - m.pos)
        return None

    def stream(self, max_ticks: int | None = None) -> Iterator[list[TelemetryMessage]]:
        n = 0
        while max_ticks is None or n < max_ticks:
            yield self.step()
            n += 1