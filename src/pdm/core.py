from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

# --- SCHEMA CONSTANTS ---
SETTING_COLUMNS = ["op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
RAW_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
INFORMATIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
SENSOR_META = {
    "s1": {"name": "T2", "desc": "Total temperature at fan inlet", "unit": "°R", "station": "Fan"},
    "s2": {"name": "T24", "desc": "Total temperature at LPC outlet", "unit": "°R", "station": "LPC"},
    "s3": {"name": "T30", "desc": "Total temperature at HPC outlet", "unit": "°R", "station": "HPC"},
    "s4": {"name": "T50", "desc": "Total temperature at LPT outlet", "unit": "°R", "station": "LPT"},
    "s5": {"name": "P2", "desc": "Pressure at fan inlet", "unit": "psia", "station": "Fan"},
    "s6": {"name": "P15", "desc": "Total pressure in bypass duct", "unit": "psia", "station": "Bypass"},
    "s7": {"name": "P30", "desc": "Total pressure at HPC outlet", "unit": "psia", "station": "HPC"},
    "s8": {"name": "Nf", "desc": "Physical fan speed", "unit": "rpm", "station": "Fan"},
    "s9": {"name": "Nc", "desc": "Physical core speed", "unit": "rpm", "station": "Core"},
    "s10": {"name": "epr", "desc": "Engine pressure ratio (P50/P2)", "unit": "-", "station": "Core"},
    "s11": {"name": "Ps30", "desc": "Static pressure at HPC outlet", "unit": "psia", "station": "HPC"},
    "s12": {"name": "phi", "desc": "Fuel flow / Ps30", "unit": "pps/psi", "station": "Combustor"},
    "s13": {"name": "NRf", "desc": "Corrected fan speed", "unit": "rpm", "station": "Fan"},
    "s14": {"name": "NRc", "desc": "Corrected core speed", "unit": "rpm", "station": "Core"},
    "s15": {"name": "BPR", "desc": "Bypass ratio", "unit": "-", "station": "Bypass"},
    "s16": {"name": "farB", "desc": "Burner fuel-air ratio", "unit": "-", "station": "Combustor"},
    "s17": {"name": "htBleed", "desc": "Bleed enthalpy", "unit": "-", "station": "HPC"},
    "s18": {"name": "Nf_dmd", "desc": "Demanded fan speed", "unit": "rpm", "station": "Fan"},
    "s19": {"name": "PCNfR_dmd", "desc": "Demanded corrected fan speed", "unit": "rpm", "station": "Fan"},
    "s20": {"name": "W31", "desc": "HPT coolant bleed", "unit": "lbm/s", "station": "HPT"},
    "s21": {"name": "W32", "desc": "LPT coolant bleed", "unit": "lbm/s", "station": "LPT"},
}

TOPIC_TEMPLATE = "factory/{site}/{line}/{machine_id}/telemetry"
TOPIC_FILTER = "factory/+/+/+/telemetry"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iso_to_epoch_ms(ts: str) -> int:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return int((dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp() * 1000)


class TelemetryMessage(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    site: str = "default"
    line: str = "default"
    ts: str = Field(default_factory=utc_now_iso)
    cycle: int = Field(ge=1)
    settings: dict[str, float]
    sensors: dict[str, float]

    @field_validator("sensors", "settings")
    @classmethod
    def _validate_keys(cls, v: dict[str, float], info: ValidationInfo) -> dict[str, float]:
        req = SENSOR_COLUMNS if info.field_name == "sensors" else SETTING_COLUMNS
        if missing := set(req) - v.keys():
            raise ValueError(f"missing {info.field_name}: {missing}")
        return v

    @property
    def topic(self) -> str:
        return TOPIC_TEMPLATE.format(site=self.site, line=self.line, machine_id=self.machine_id)

    @property
    def ts_ms(self) -> int:
        return iso_to_epoch_ms(self.ts)

    def raw_vector(self) -> list[float]:
        return [self.settings[c] for c in SETTING_COLUMNS] + [self.sensors[c] for c in SENSOR_COLUMNS]


# --- CONFIGURATION ---
def env(name: str, default, cast_type=str):
    return cast_type(os.environ.get(name, default))


@dataclass(frozen=True)
class HealthThresholds:
    rul_cap: float = field(default_factory=lambda: env("PDM_RUL_CAP", 125.0, float))
    rul_warning: float = field(default_factory=lambda: env("PDM_RUL_WARNING", 50.0, float))
    rul_critical: float = field(default_factory=lambda: env("PDM_RUL_CRITICAL", 20.0, float))
    anomaly_warning: float = field(default_factory=lambda: env("PDM_ANOMALY_WARNING", 1.0, float))
    anomaly_critical: float = field(default_factory=lambda: env("PDM_ANOMALY_CRITICAL", 2.0, float))
    safety_margin_cycles: float = field(default_factory=lambda: env("PDM_SAFETY_MARGIN_CYCLES", 10.0, float))
    hysteresis_rul: float = field(default_factory=lambda: env("PDM_HYSTERESIS_RUL", 8.0, float))
    hysteresis_anomaly: float = field(default_factory=lambda: env("PDM_HYSTERESIS_ANOMALY", 0.25, float))
    anomaly_ewma_alpha: float = field(default_factory=lambda: env("PDM_ANOMALY_EWMA_ALPHA", 0.3, float))


@dataclass(frozen=True)
class PlantEconomics:
    cycle_hours: float = field(default_factory=lambda: env("PDM_CYCLE_HOURS", 8.0, float))
    downtime_cost_per_hour: float = field(
        default_factory=lambda: env("PDM_DOWNTIME_COST_PER_HOUR", 15_000.0, float)
    )
    unplanned_outage_hours: float = field(default_factory=lambda: env("PDM_UNPLANNED_OUTAGE_HOURS", 24.0, float))
    planned_outage_hours: float = field(default_factory=lambda: env("PDM_PLANNED_OUTAGE_HOURS", 6.0, float))
    maintenance_crews: int = field(default_factory=lambda: env("PDM_MAINTENANCE_CREWS", 2, int))


@dataclass(frozen=True)
class Settings:
    backend: str = field(default_factory=lambda: env("PDM_BACKEND", "local"))
    # FIX: was "ap-northeast1" (invalid region name).
    aws_region: str = field(
        default_factory=lambda: env("AWS_REGION", env("AWS_DEFAULT_REGION", "ap-northeast-1"))
    )
    telemetry_table: str = field(default_factory=lambda: env("PDM_TELEMETRY_TABLE", "pdm-telemetry"))
    machine_state_table: str = field(default_factory=lambda: env("PDM_MACHINE_STATE_TABLE", "pdm-machine-state"))
    alerts_table: str = field(default_factory=lambda: env("PDM_ALERTS_TABLE", "pdm-alerts"))
    local_db_path: str = field(default_factory=lambda: env("PDM_LOCAL_DB", "artifacts/pdm_local.sqlite"))
    model_dir: str = field(default_factory=lambda: env("PDM_MODEL_DIR", "artifacts/model"))
    model_s3_uri: str = field(default_factory=lambda: env("PDM_MODEL_S3_URI", ""))
    firehose_stream: str = field(default_factory=lambda: env("PDM_FIREHOSE_STREAM", ""))
    sns_topic_arn: str = field(default_factory=lambda: env("PDM_SNS_TOPIC_ARN", ""))
    telemetry_ttl_days: int = field(default_factory=lambda: env("PDM_TELEMETRY_TTL_DAYS", 7, int))
    thresholds: HealthThresholds = field(default_factory=HealthThresholds)
    economics: PlantEconomics = field(default_factory=PlantEconomics)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (the defaults live in the field factories)."""
        return cls()