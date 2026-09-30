from __future__ import annotations

import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pdm.core import SENSOR_COLUMNS, SENSOR_META, Settings
from pdm.storage import get_store

# --- CONSTANTS & STYLING ---
STATUS_COLORS = {"healthy": "#22c55e", "warning": "#f59e0b", "critical": "#ef4444", "unknown": "#64748b"}
STATUS_EMOJI = {"healthy": "🟢", "warning": "🟠", "critical": "🔴", "unknown": "⚪"}
SEVERITY_COLORS = {"critical": "#ef4444", "warning": "#f59e0b", "info": "#38bdf8"}
REFRESH_SECONDS = int(os.environ.get("PDM_DASHBOARD_REFRESH_SECONDS", "5"))

# --- CORE CONFIG ---
@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    return Settings.from_env()

@st.cache_resource(show_spinner=False)
def store():
    return get_store(settings())

@st.cache_resource(show_spinner=False)
def model_bundle():
    try:
        from pdm.engine import load_bundle
        s = settings()
        return load_bundle(s.model_dir, s.model_s3_uri)
    except Exception:
        return None

# --- DATA LOADERS ---
def load_states() -> pd.DataFrame:
    rows = store().list_machine_states()
    if not rows:
        return pd.DataFrame(columns=["machine_id", "site", "line", "status", "health_index", "rul_p10", "rul_p50", "rul_p90", "anomaly_score", "cycle", "maintenance_due", "last_ts", "reasons"])
    df = pd.DataFrame(rows)
    df["maintenance_due_dt"] = pd.to_datetime(df["maintenance_due"], utc=True, errors="coerce")
    df["last_seen"] = pd.to_datetime(df["last_ts"], unit="ms", utc=True, errors="coerce")
    df["status"] = df["status"].fillna("unknown")
    return df.sort_values("machine_id").reset_index(drop=True)

def load_telemetry(machine_id: str, limit: int = 400) -> pd.DataFrame:
    rows = store().get_telemetry(machine_id, limit=limit)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = pd.concat([df.drop(columns=["sensors", "settings"], errors="ignore"), pd.DataFrame(list(df["sensors"]))], axis=1)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def load_alerts(status: str | None = None, limit: int = 300) -> pd.DataFrame:
    rows = store().list_alerts(status=status, limit=limit)
    if not rows:
        return pd.DataFrame(columns=["machine_id", "ts", "severity", "type", "status", "message", "cycle", "rul_p50", "anomaly_score"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts", ascending=False).reset_index(drop=True)

# --- FORMATTING HELPERS ---
def sensor_label(col: str) -> str:
    return f"{col} · {SENSOR_META[col]['name']} ({SENSOR_META[col]['unit']})" if col in SENSOR_META else col

def sensor_options() -> list[str]:
    return list(SENSOR_COLUMNS)

def fmt_dt(dt) -> str:
    return "–" if dt is None or pd.isna(dt) else pd.to_datetime(dt, utc=True).strftime("%Y-%m-%d %H:%M UTC")

def badge(status: str) -> str:
    return f'<span class="pdm-badge" style="background:{STATUS_COLORS.get(status, STATUS_COLORS["unknown"])}">{status.upper()}</span>'

def humanize_delta(dt) -> str:
    if dt is None or pd.isna(dt):
        return "–"
    secs = abs(((dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt) - datetime.now(UTC)).total_seconds())
    sign = "" if secs >= 0 else "-"
    if secs < 3600:
        return f"{sign}{int(secs // 60)} min"
    if secs < 86400:
        return f"{sign}{secs / 3600:.1f} h"
    return f"{sign}{secs / 86400:.1f} d"

def machine_card(row: pd.Series) -> str:
    status, health, due = row.get("status", "unknown"), float(row.get("health_index", 0) or 0), row.get("maintenance_due_dt")
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    return f"""
<div class="pdm-card {status}">
  <h4>{row["machine_id"]} {badge(status)}</h4>
  <div class="sub">{row.get("site", "")} / {row.get("line", "")} · cycle {int(row.get("cycle", 0))} · seen {humanize_delta(row.get("last_seen"))} ago</div>
  <div class="row"><span>Health</span><b>{health:.0f}%</b></div>
  <div class="row"><span>RUL p50</span><b>{row["rul_p50"]:.0f} <span class="pdm-muted">({row["rul_p10"]:.0f}–{row["rul_p90"]:.0f})</span></b></div>
  <div class="row"><span>Anomaly</span><b>{row["anomaly_score"]:.2f}</b></div>
  <div class="row"><span>Maintain by</span><b>{humanize_delta(due)}</b></div>
  <div class="pdm-bar"><div style="width:{max(2, health):.0f}%; background:{color}"></div></div>
</div>"""

# --- PLOTLY CHARTS ---
PLOT_BG, GRID, FONT = "rgba(0,0,0,0)", "#1e293b", "#e2e8f0"

def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT, size=12), margin=dict(l=10, r=10, t=36, b=10), height=height, legend=dict(orientation="h", y=1.08, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig

def plant_floor(states: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if states.empty:
        return _layout(fig, 260, title="Plant floor")
    lines = sorted(states["line"].unique())
    for i, line in enumerate(lines):
        y = -i * 1.0
        fig.add_shape(type="rect", x0=-0.6, x1=max(6, states["line"].value_counts().max()) - 0.4, y0=y - 0.42, y1=y + 0.42, line=dict(color=GRID), fillcolor="rgba(30,41,59,0.45)", layer="below")
        fig.add_annotation(x=-0.55, y=y + 0.32, text=f"<b>Line {line}</b>", showarrow=False, xanchor="left", font=dict(size=11, color="#94a3b8"))
        sub = states[states["line"] == line].reset_index(drop=True)
        fig.add_trace(go.Scatter(x=sub.index, y=[y] * len(sub), mode="markers+text", text=sub["machine_id"], textposition="bottom center", textfont=dict(size=10, color=FONT), marker=dict(size=18 + 22 * (sub["health_index"].fillna(0) / 100.0), color=[STATUS_COLORS.get(s, STATUS_COLORS["unknown"]) for s in sub["status"]], line=dict(width=2, color="#0b1220"), symbol="hexagon"), hovertemplate="<b>%{text}</b><br>health: %{customdata[1]:.0f}%<extra></extra>", customdata=np.stack([sub["status"], sub["health_index"]], axis=1), showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-(len(lines) - 1) * 1.0 - 0.7, 0.7])
    return _layout(fig, 90 + 95 * len(lines), title="Plant floor · digital twin")

def fleet_rul_bars(states: pd.DataFrame, warn: float, crit: float) -> go.Figure:
    fig = go.Figure()
    if states.empty:
        return _layout(fig, 300, title="Remaining useful life")
    df = states.sort_values("rul_p50")
    fig.add_trace(go.Bar(x=df["rul_p50"], y=df["machine_id"], orientation="h", marker_color=[STATUS_COLORS.get(s, "#64748b") for s in df["status"]], error_x=dict(type="data", array=(df["rul_p90"] - df["rul_p50"]).clip(lower=0), arrayminus=(df["rul_p50"] - df["rul_p10"]).clip(lower=0), color="#94a3b8", thickness=1.2), hovertemplate="<b>%{y}</b><br>RUL p50 %{x:.0f} cycles<extra></extra>"))
    fig.add_vline(x=crit, line=dict(color=STATUS_COLORS["critical"], dash="dot"), annotation_text="critical")
    fig.add_vline(x=warn, line=dict(color=STATUS_COLORS["warning"], dash="dot"), annotation_text="warning")
    return _layout(fig, max(260, 26 * len(df) + 80), title="Remaining useful life by machine")

def rul_trajectory(tele: pd.DataFrame, warn: float, crit: float, cap: float, x: str = "cycle") -> go.Figure:
    fig = go.Figure()
    if tele.empty:
        return _layout(fig, 320, title="RUL prediction")
    fig.add_trace(go.Scatter(x=tele[x], y=tele["rul_p90"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=tele[x], y=tele["rul_p10"], fill="tonexty", fillcolor="rgba(56,189,248,0.18)", line=dict(width=0), name="p10–p90 band"))
    fig.add_trace(go.Scatter(x=tele[x], y=tele["rul_p50"], name="RUL p50", line=dict(color="#38bdf8", width=2.5)))
    fig.add_trace(go.Scatter(x=tele[x], y=tele["health_index"], name="health index (%)", yaxis="y2", line=dict(color="#a78bfa", width=1.5, dash="dot")))
    fig.add_hline(y=warn, line=dict(color=STATUS_COLORS["warning"], dash="dot"))
    fig.add_hline(y=crit, line=dict(color=STATUS_COLORS["critical"], dash="dot"))
    fig.update_layout(yaxis=dict(title="RUL (cycles)", range=[0, cap * 1.05]), yaxis2=dict(title="health %", overlaying="y", side="right", range=[0, 105], showgrid=False))
    return _layout(fig, 340, title="Remaining useful life · prediction over time")

def anomaly_timeline(tele: pd.DataFrame, warn: float, crit: float, x: str = "cycle") -> go.Figure:
    fig = go.Figure()
    if tele.empty:
        return _layout(fig, 260, title="Anomaly score")
    colors = np.where(tele["anomaly_score"] >= crit, STATUS_COLORS["critical"], np.where(tele["anomaly_score"] >= warn, STATUS_COLORS["warning"], "#38bdf8"))
    fig.add_trace(go.Bar(x=tele[x], y=tele["anomaly_score"], marker_color=colors, name="anomaly score"))
    fig.add_hline(y=warn, line=dict(color=STATUS_COLORS["warning"], dash="dot"), annotation_text="warning")
    fig.add_hline(y=crit, line=dict(color=STATUS_COLORS["critical"], dash="dot"), annotation_text="critical")
    return _layout(fig, 260, title="Anomaly score · distance from healthy baseline")

def health_gauge(value: float, status: str) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number=dict(suffix="%", font=dict(size=30)), gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94a3b8"), bar=dict(color=STATUS_COLORS.get(status, "#64748b"), thickness=0.3), bgcolor="rgba(0,0,0,0)", borderwidth=0, steps=[dict(range=[0, 30], color="rgba(239,68,68,0.18)"), dict(range=[30, 60], color="rgba(245,158,11,0.18)"), dict(range=[60, 100], color="rgba(34,197,94,0.18)")]), title=dict(text="Health index", font=dict(size=13))))
    return _layout(fig, 210, margin=dict(l=20, r=20, t=40, b=0))

_STATION_X = {"Fan": 0.9, "Bypass": 1.6, "LPC": 2.3, "HPC": 3.4, "Combustor": 4.5, "HPT": 5.3, "LPT": 6.2, "Core": 3.9}

def engine_twin(latest: pd.Series, zscores: dict[str, float], sensors: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.3, x1=7.4, y0=-1.6, y1=1.6, line=dict(color="#334155", width=2), fillcolor="rgba(30,41,59,0.35)", layer="below")
    fig.add_shape(type="rect", x0=1.9, x1=6.9, y0=-0.75, y1=0.75, line=dict(color="#475569", width=1.5), fillcolor="rgba(51,65,85,0.5)", layer="below")
    fig.add_shape(type="line", x0=0.7, x1=6.8, y0=0, y1=0, line=dict(color="#94a3b8", width=3))
    slots = {}
    for s in sensors:
        meta = SENSOR_META.get(s, {"station": "Core", "name": s, "unit": ""})
        x, k = _STATION_X.get(meta["station"], 3.9), slots.get(meta["station"], 0)
        slots[meta["station"]] = k + 1
        side, level, z, val = 1 if k % 2 == 0 else -1, 2.1 + 0.45 * (k // 2), zscores.get(s, 0.0), latest.get(s, np.nan)
        color = STATUS_COLORS["critical"] if abs(z) >= 3 else STATUS_COLORS["warning"] if abs(z) >= 2 else "#cbd5e1"
        fig.add_annotation(x=x, y=side * level, ax=x, ay=side * 1.65, axref="x", ayref="y", text=f"<b>{meta['name']}</b> {val:,.2f} {meta['unit']}<br><span style='font-size:10px'>z={z:+.1f}</span>", showarrow=True, arrowhead=0, arrowcolor="#475569", font=dict(size=10, color=color), bgcolor="rgba(15,23,42,0.9)", bordercolor=color, borderwidth=1, borderpad=3)
    ymax = 2.1 + 0.45 * (max(slots.values(), default=1) // 2) + 0.6
    fig.update_xaxes(visible=False, range=[0, 7.7])
    fig.update_yaxes(visible=False, range=[-ymax, ymax], scaleanchor="x", scaleratio=1)
    return _layout(fig, 430, title="Asset twin · live read-outs")

def maintenance_gantt(plan: pd.DataFrame) -> go.Figure:
    if plan.empty:
        return _layout(go.Figure(), 260, title="Maintenance plan")
    fig = px.timeline(plan, x_start="start", x_end="end", y="machine_id", color="priority", color_discrete_map={"P1": STATUS_COLORS["critical"], "P2": STATUS_COLORS["warning"], "P3": STATUS_COLORS["healthy"]})
    fig.add_trace(go.Scatter(x=plan["due"], y=plan["machine_id"], mode="markers", name="deadline", marker=dict(symbol="line-ns-open", size=16, color="#e2e8f0", line=dict(width=2))))
    fig.update_yaxes(autorange="reversed", title="")
    return _layout(fig, max(280, 28 * len(plan) + 100), title="Capacity-aware maintenance plan")