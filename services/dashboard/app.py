"""Predictive Maintenance & Digital Twin console (Streamlit)."""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from common import (  # noqa: E402
    REFRESH_SECONDS,
    STATUS_EMOJI,
    anomaly_timeline,
    badge,
    engine_twin,
    fleet_rul_bars,
    fmt_dt,
    health_gauge,
    load_alerts,
    load_states,
    load_telemetry,
    machine_card,
    maintenance_gantt,
    model_bundle,
    plant_floor,
    rul_trajectory,
    sensor_label,
    settings,
    store,
)
from pdm.core import INFORMATIVE_SENSORS  # noqa: E402
from pdm.engine import maintenance_due, roi_estimate, schedule_maintenance  # noqa: E402

# ---------------------------------------------------------------------------
# Styling / auth
# ---------------------------------------------------------------------------
CSS = """
<style>
:root { --card-bg: #0f172a; --card-border: #1e293b; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.pdm-card { background: var(--card-bg); border: 1px solid var(--card-border); border-left: 6px solid #64748b; border-radius: 10px; padding: 0.8rem 0.9rem; margin-bottom: 0.6rem; min-height: 150px; }
.pdm-card.healthy { border-left-color: #22c55e; } .pdm-card.warning { border-left-color: #f59e0b; }
.pdm-card.critical { border-left-color: #ef4444; animation: pdm-pulse 1.6s ease-in-out infinite; }
@keyframes pdm-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.0);} 50% { box-shadow: 0 0 0 4px rgba(239,68,68,0.25);} }
.pdm-card h4 { margin: 0 0 0.25rem 0; font-size: 1.05rem; }
.pdm-card .sub { color: #94a3b8; font-size: 0.78rem; margin-bottom: 0.35rem; }
.pdm-card .row { display: flex; justify-content: space-between; font-size: 0.86rem; margin: 0.12rem 0; }
.pdm-card .row b { font-variant-numeric: tabular-nums; }
.pdm-bar { height: 7px; background: #1e293b; border-radius: 4px; overflow: hidden; margin-top: 0.4rem; }
.pdm-bar > div { height: 100%; }
.pdm-badge { display:inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; color: #0b1220; }
.pdm-muted { color:#94a3b8; font-size:0.8rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def require_login() -> None:
    """Optional password gate (set DASHBOARD_PASSWORD to enable)."""
    pw = os.environ.get("DASHBOARD_PASSWORD", "")
    if not pw or st.session_state.get("authed"):
        return
    st.title("🔐 Predictive Maintenance Console")
    with st.form("login"):
        code = st.text_input("Access code", type="password")
        if st.form_submit_button("Enter"):
            if code == pw:
                st.session_state["authed"] = True
                st.rerun()
            st.error("Invalid access code")
    st.stop()


# Shortcuts to config used by several pages
TH = settings().thresholds
ECON = settings().economics


# ---------------------------------------------------------------------------
# Page 1: Fleet overview
# ---------------------------------------------------------------------------
def page_fleet() -> None:
    st.title("🏭 Fleet overview")
    st.caption("Live digital-twin view of every monitored asset.")
    sort_by = st.radio("Sort cards by", ["risk", "machine_id"], horizontal=True, label_visibility="collapsed")

    @st.fragment(run_every=f"{REFRESH_SECONDS}s")
    def live() -> None:
        states = load_states()
        if states.empty:
            st.warning("No machine state yet. Start the simulator and scorer.", icon="📡")
            return
        open_alerts = load_alerts(status="open", limit=500)

        # KPI strip
        n_crit = int((states["status"] == "critical").sum())
        n_warn = int((states["status"] == "warning").sum())
        at_risk = n_crit + n_warn
        exposure = at_risk * (ECON.unplanned_outage_hours - ECON.planned_outage_hours) * ECON.downtime_cost_per_hour
        k = st.columns(6)
        k[0].metric("Machines online", len(states))
        k[1].metric("Fleet health", f"{states['health_index'].mean():.0f}%")
        k[2].metric("🔴 Critical", n_crit)
        k[3].metric("🟠 Warning", n_warn)
        k[4].metric("Open alerts", len(open_alerts))
        k[5].metric("Downtime cost at risk*", f"${exposure:,.0f}")
        st.caption("*at-risk machines × (unplanned − planned outage hours) × downtime cost/hour")

        # Charts
        left, right = st.columns([3, 2])
        left.plotly_chart(plant_floor(states), use_container_width=True)
        right.plotly_chart(fleet_rul_bars(states, TH.rul_warning, TH.rul_critical), use_container_width=True)

        # Machine cards
        st.subheader("Machines")
        if sort_by == "risk":
            states = states.sort_values(["health_index", "rul_p50"])
        cols = st.columns(4)
        for i, (_, row) in enumerate(states.iterrows()):
            cols[i % 4].markdown(machine_card(row), unsafe_allow_html=True)

    live()


# ---------------------------------------------------------------------------
# Page 2: Machine twin
# ---------------------------------------------------------------------------
def page_machine() -> None:
    st.title("⚙️ Machine twin")
    states = load_states()
    if states.empty:
        st.warning("No machines reporting yet.", icon="📡")
        st.stop()

    c1, c2 = st.columns([2, 3])
    machine_id = c1.selectbox("Machine", states["machine_id"].tolist(), key="selected_machine")
    sensors = c2.multiselect(
        "Sensors on the twin",
        INFORMATIVE_SENSORS,
        default=INFORMATIVE_SENSORS[:8],
        format_func=sensor_label,
    )

    @st.fragment(run_every=f"{REFRESH_SECONDS}s")
    def live() -> None:
        row = load_states().set_index("machine_id").loc[machine_id]
        tele = load_telemetry(machine_id, limit=300)

        st.markdown(f"## {machine_id} {badge(row['status'])}", unsafe_allow_html=True)
        st.caption(f"{row['site']} / {row['line']} · cycle {int(row['cycle'])} · last seen {fmt_dt(row['last_seen'])}")

        m = st.columns(5)
        m[0].metric("Health index", f"{row['health_index']:.0f}%")
        m[1].metric("RUL p50", f"{row['rul_p50']:.0f} cycles")
        m[2].metric("RUL p10–p90", f"{row['rul_p10']:.0f}–{row['rul_p90']:.0f}")
        m[3].metric("Anomaly score", f"{row['anomaly_score']:.2f}")
        m[4].metric("Maintain by", fmt_dt(row["maintenance_due_dt"]))

        reasons = row.get("reasons") or []
        if reasons:
            st.info(" · ".join(reasons), icon="ℹ️")

        if tele.empty:
            st.warning("No telemetry history yet for this machine.")
            return

        a, b = st.columns([3, 2])
        a.plotly_chart(rul_trajectory(tele, TH.rul_warning, TH.rul_critical, TH.rul_cap), use_container_width=True)
        b.plotly_chart(health_gauge(float(row["health_index"]), row["status"]), use_container_width=True)
        b.plotly_chart(anomaly_timeline(tele, TH.anomaly_warning, TH.anomaly_critical), use_container_width=True)

        # Engine twin: z-score of latest reading vs. this machine's early history
        baseline = tele.head(max(5, len(tele) // 3))
        latest = tele.iloc[-1]
        zscores = {
            s: float((latest[s] - baseline[s].mean()) / (baseline[s].std() or 1.0))
            for s in sensors if s in tele.columns
        }
        st.plotly_chart(engine_twin(latest, zscores, sensors), use_container_width=True)

        with st.expander("Raw sensor trends"):
            pick = st.multiselect("Sensors", sensors, default=sensors[:3], format_func=sensor_label, key="raw_pick")
            if pick:
                st.line_chart(tele.set_index("cycle")[pick])

    live()


# ---------------------------------------------------------------------------
# Page 3: Alerts
# ---------------------------------------------------------------------------
def page_alerts() -> None:
    st.title("🚨 Alerts")
    st.caption("Alerts fire on status transitions and recoveries.")

    f1, f2, f3 = st.columns(3)
    status_filter = f1.selectbox("Status", ["open", "acknowledged", "all"])
    severity_filter = f2.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning"])
    machine_filter = f3.text_input("Machine contains", "")

    @st.fragment(run_every=f"{REFRESH_SECONDS}s")
    def live() -> None:
        df = load_alerts(status=None if status_filter == "all" else status_filter, limit=500)
        if severity_filter and not df.empty:
            df = df[df["severity"].isin(severity_filter)]
        if machine_filter and not df.empty:
            df = df[df["machine_id"].str.contains(machine_filter, case=False)]
        if df.empty:
            st.success("No alerts match the current filter.", icon="✅")
            return

        c = st.columns(4)
        c[0].metric("Shown", len(df))
        c[1].metric("🔴 Critical", int((df["severity"] == "critical").sum()))
        c[2].metric("🟠 Warning", int((df["severity"] == "warning").sum()))
        c[3].metric("Machines affected", df["machine_id"].nunique())

        show = df.assign(severity=df["severity"].map(lambda s: f"{STATUS_EMOJI.get(s, '🔵')} {s}"))
        cols = [c for c in ["time", "machine_id", "severity", "type", "status", "message",
                            "cycle", "rul_p50", "anomaly_score"] if c in show.columns]
        event = st.dataframe(show[cols], hide_index=True, on_select="rerun", selection_mode="multi-row",
                             use_container_width=True)

        selected = df.iloc[event.selection.rows] if event.selection.rows else df.iloc[0:0]
        if st.button(f"✅ Acknowledge selected ({len(selected)})",
                     disabled=selected.empty or status_filter == "acknowledged"):
            for _, a in selected.iterrows():
                store().ack_alert(a["machine_id"], int(a["ts"]), user="console")
            st.rerun()

    live()


# ---------------------------------------------------------------------------
# Page 4: Maintenance & ROI
# ---------------------------------------------------------------------------
def page_maintenance() -> None:
    st.title("🗓️ Maintenance schedule & business case")
    st.caption("Converts RUL p10 into deadlines, packs them onto crews, and estimates ROI.")

    states = load_states()
    now = datetime.now(UTC)

    st.subheader("Recommended plan")
    if states.empty:
        st.warning("No machines reporting yet.", icon="📡")
    else:
        machines = [
            {
                "machine_id": r["machine_id"],
                "status": r["status"],
                "rul_p10": float(r["rul_p10"]),
                "maintenance_due": maintenance_due(now, float(r["rul_p10"]), TH, ECON)[0],
            }
            for _, r in states.iterrows()
        ]
        plan = pd.DataFrame(schedule_maintenance(machines, now, ECON))
        st.plotly_chart(maintenance_gantt(plan), use_container_width=True)
        st.dataframe(
            plan[["priority", "machine_id", "status", "rul_p10", "crew", "start", "end", "due", "slack_hours"]],
            hide_index=True, use_container_width=True,
        )
        st.caption(f"{ECON.maintenance_crews} crew(s) · {ECON.planned_outage_hours:.0f} h per planned job")

    st.divider()
    st.subheader("ROI estimate")
    c1, c2, c3 = st.columns(3)
    n_machines = c1.number_input("Machines in scope", 1, 10_000, max(1, len(states)))
    events = c2.slider("Unplanned failures / machine / year", 0.0, 5.0, 1.0, 0.1)
    capture = c3.slider("Share caught by PdM", 0.0, 1.0, 0.7, 0.05)
    cost = st.number_input("Platform cost per year ($)", 0.0, 10_000_000.0, 120_000.0, 1_000.0)

    roi = roi_estimate(int(n_machines), events, capture, ECON, cost)
    k = st.columns(4)
    k[0].metric("Gross saving / yr", f"${roi['gross_saving']:,.0f}")
    k[1].metric("Net benefit / yr", f"${roi['net_benefit']:,.0f}")
    k[2].metric("ROI", f"{roi['roi_pct']:,.0f}%" if roi["roi_pct"] != float("inf") else "∞")
    k[3].metric("Payback", f"{roi['payback_months']:.1f} months" if roi["payback_months"] != float("inf") else "–")
    st.caption(
        f"Assumes ${ECON.downtime_cost_per_hour:,.0f}/h downtime, "
        f"{ECON.unplanned_outage_hours:.0f} h unplanned vs {ECON.planned_outage_hours:.0f} h planned outage. "
        "Estimates depend entirely on these inputs."
    )


# ---------------------------------------------------------------------------
# Page 5: Model card
# ---------------------------------------------------------------------------
def page_model() -> None:
    st.title("🧠 Model card")
    bundle = model_bundle()
    if bundle is None:
        st.warning("Model bundle not available. Run `make train` first.")
        st.stop()

    meta = bundle.metadata
    metrics = meta.get("metrics", {})

    st.markdown(
        f"""
**Task.** Predict remaining useful life (RUL) and flag anomalous behaviour of turbofan engines.

**Data.** NASA C-MAPSS `{meta.get('dataset', '?')}` · rolling window of **{bundle.window}** cycles ·
RUL capped at **{bundle.rul_cap:.0f}** · trained {meta.get('trained_at', '–')[:19]}.

**Models.** LightGBM regressor (RUL p50) · bucketed conformal offsets (p10/p90) ·
Mahalanobis anomaly detector fitted on early-life windows.
"""
    )

    if metrics:
        k = st.columns(4)
        k[0].metric("Test RMSE (capped)", f"{metrics.get('test_rmse_capped', 0):.1f} cycles")
        k[1].metric("NASA PHM08 score", f"{metrics.get('test_nasa_score_capped', 0):,.0f}")
        k[2].metric("p10–p90 coverage", f"{metrics.get('test_coverage_p10_p90', 0) * 100:.0f}%")
        k[3].metric("Mean interval width", f"{metrics.get('test_mean_interval_width', 0):.0f} cycles")

        top = metrics.get("top_features_gain", [])
        if top:
            st.subheader("Top features (gain)")
            st.bar_chart(pd.DataFrame(top).set_index("feature")["gain"])

        with st.expander("All metrics"):
            st.json(metrics)

    st.subheader("Limitations")
    st.markdown(
        """
- C-MAPSS is simulated data; results do not transfer to real industrial equipment.
- Evaluated on a single sub-dataset (FD001, one operating condition, one fault mode).
- Prediction intervals are empirical (conformal on held-out engines), not guarantees.
- Anomaly scores measure distance from an early-life baseline, not confirmed faults.
"""
    )


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="PdM Digital Twin Console", page_icon="🏭",
        layout="wide", initial_sidebar_state="expanded",
    )
    inject_css()
    require_login()

    pages = [
        st.Page(page_fleet, title="Fleet overview", icon="🏭", default=True),
        st.Page(page_machine, title="Machine twin", icon="⚙️"),
        st.Page(page_alerts, title="Alerts", icon="🚨"),
        st.Page(page_maintenance, title="Maintenance & ROI", icon="🗓️"),
        st.Page(page_model, title="Model card", icon="🧠"),
    ]

    with st.sidebar:
        st.markdown("### 🏭 PdM Digital Twin")
        s = settings()
        backend = "AWS DynamoDB" if s.backend == "dynamodb" else "local demo store"
        st.caption(f"Data source: **{backend}** · region `{s.aws_region}`")
        st.caption(f"Auto-refresh every {REFRESH_SECONDS}s")
        if os.environ.get("PDM_DEMO_MODE"):
            st.info("Demo mode: an in-process simulator is replaying NASA C-MAPSS engines through the scorer.", icon="🧪")

    st.navigation(pages).run()