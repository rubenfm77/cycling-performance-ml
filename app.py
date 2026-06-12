# app.py
"""
Cycling Performance Dashboard — Streamlit
==========================================
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Cycling Performance",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main { background-color: #0d1117; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stSidebar { background-color: #161b22; }
    div[data-testid="stMetricValue"] { color: #58a6ff; font-size: 2rem; }
    div[data-testid="stMetricDelta"] { font-size: 0.9rem; }
    .state-box { border-radius: 8px; padding: 12px 20px;
                 text-align: center; font-weight: bold; font-size: 1.1rem; }
    .week-card { background: #161b22; border: 1px solid #30363d;
                 border-radius: 10px; padding: 16px; margin: 4px; }
</style>
""", unsafe_allow_html=True)

C = {
    "bg": "#0d1117", "panel": "#161b22", "grid": "#21262d",
    "text": "#c9d1d9", "muted": "#8b949e", "accent": "#58a6ff",
    "green": "#3fb950", "orange": "#f0883e", "red": "#f85149",
    "yellow": "#d29922", "purple": "#bc8cff",
}

# ── Athlete constants — update here when FTP changes ─────────────────────────
FTP_CURRENT = 240          # validated 10 Jun 2026 — 4x10min @ 240-242W, stable HR
FTP_PRE     = 275          # pre-accident reference
WEIGHT_KG   = 57.0
WKG_CURRENT = FTP_CURRENT / WEIGHT_KG   # ~4.21 W/kg
WKG_PRE     = FTP_PRE / WEIGHT_KG       # ~4.82 W/kg

STATE_COLORS = {
    "Undertrained": C["muted"],
    "Overreached": C["red"],
    "Deep Block": C["orange"],
    "Build Phase": C["yellow"],
    "Neutral": C["accent"],
    "Fresh": C["green"],
    "Peak/Detrain Risk": C["purple"],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
    font=dict(color=C["text"]),
    xaxis=dict(gridcolor=C["grid"], color=C["muted"]),
    yaxis=dict(gridcolor=C["grid"], color=C["muted"]),
    margin=dict(l=40, r=20, t=40, b=40),
)

MAIN_TYPES = [
    "END", "AEROBIC BASE", "FTP", "FATMAX", "VO2MAX",
    "SST", "TEMPO", "TORQUE", "Q-I INTERVALS", "BILLAT", "PIRAMIDAL"
]

FTP_DRIVERS = ["FTP", "SST", "TEMPO", "PIRAMIDAL", "VO2MAX", "BILLAT", "Q-I INTERVALS"]
BASE_TYPES  = ["END", "AEROBIC BASE", "FATMAX", "TORQUE"]


def fetch_from_intervals_icu():
    """Fetch activities directly from the Intervals.icu API.

    Used on Streamlit Cloud, where the local data/ folder doesn't exist.
    Requires two values in the app's Secrets:
        INTERVALS_ATHLETE_ID = "i123456"
        INTERVALS_API_KEY    = "your-api-key"
    """
    try:
        athlete_id = st.secrets["INTERVALS_ATHLETE_ID"]
        api_key    = st.secrets["INTERVALS_API_KEY"]
    except Exception:
        return None

    import requests
    try:
        r = requests.get(
            f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities",
            params={"oldest": "2019-01-01", "newest": "2031-12-31"},
            auth=("API_KEY", api_key),
            timeout=30,
        )
        r.raise_for_status()
        acts = r.json()
    except Exception as e:
        st.error(f"Intervals.icu API error: {e}")
        return None
    if not acts:
        return None

    raw = pd.DataFrame(acts)
    if "type" in raw.columns:
        rides = raw[raw["type"].isin(["Ride", "VirtualRide", "GravelRide"])]
        if len(rides) > 0:
            raw = rides

    def g(col):
        return (pd.to_numeric(raw[col], errors="coerce")
                if col in raw.columns
                else pd.Series(np.nan, index=raw.index))

    df = pd.DataFrame(index=raw.index)
    df["date"] = pd.to_datetime(
        raw["start_date_local"] if "start_date_local" in raw.columns
        else raw.get("start_date"), errors="coerce"
    ).dt.tz_localize(None)
    df["tss"] = g("icu_training_load")
    intensity = g("icu_intensity")
    if intensity.notna().sum() > 0 and intensity.median(skipna=True) > 2:
        intensity = intensity / 100.0
    df["if_score"]   = intensity
    pwr = g("icu_average_watts")
    if pwr.notna().sum() == 0:
        pwr = g("average_watts")
    df["power_avg"]  = pwr
    df["duration_h"] = g("moving_time") / 3600.0
    df["hr_avg"]     = g("average_heartrate")
    df["elevation"]  = g("total_elevation_gain")
    df["weight"]     = g("icu_weight")
    df["eftp"]       = g("icu_eftp")
    df["icu_ctl"]    = g("icu_fitness")
    df["icu_atl"]    = g("icu_fatigue")
    for zc in [f"z{i}_secs" for i in range(1, 8)]:
        df[zc] = g(zc)

    names = (raw["name"] if "name" in raw.columns
             else pd.Series([""] * len(raw), index=raw.index))
    names = names.fillna("").astype(str).str.upper()

    def map_type(n):
        for t in ["PIRAMIDAL", "VO2MAX", "FATMAX", "BILLAT", "TORQUE",
                  "TEMPO", "SST", "FTP", "Q-I", "AEROBIC BASE", "END"]:
            if t in n:
                return "Q-I INTERVALS" if t == "Q-I" else t
        return "AEROBIC BASE"

    df["training_type"] = [map_type(n) for n in names]
    return df.dropna(subset=["date"]).reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_data():
    paths = [
        "data/combined_training_data.csv",
        "data/JOIN_STRAVA_TP.xlsx",
    ]
    df = None
    for p in paths:
        if Path(p).exists():
            if p.endswith(".csv"):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
                df = df.rename(columns={
                    "Activity Date": "date",
                    "TRAINING_TYPE": "training_type",
                    "TSS": "tss",
                    "IF": "if_score",
                    "PowerAverage": "power_avg",
                    "TimeTotalInHours": "duration_h",
                    "HeartRateAverage": "hr_avg",
                    "WEIGHT_KG": "weight",
                    "Elevation Gain": "elevation",
                    "icu_eftp": "eftp",
                    "icu_fitness": "icu_ctl",
                    "icu_fatigue": "icu_atl",
                })
            break

    if df is None:
        df = fetch_from_intervals_icu()

    if df is None:
        st.error(
            "No data found. Locally: run `python src/intervals_api.py` first. "
            "On Streamlit Cloud: add INTERVALS_ATHLETE_ID and "
            "INTERVALS_API_KEY in the app Secrets."
        )
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for src_col, dst_col in [("icu_eftp", "eftp"),
                             ("icu_fitness", "icu_ctl"),
                             ("icu_fatigue", "icu_atl")]:
        if dst_col not in df.columns and src_col in df.columns:
            df[dst_col] = df[src_col]

    df["tss"] = pd.to_numeric(df["tss"], errors="coerce").fillna(0)
    df["ctl"] = df["tss"].ewm(span=42, adjust=False).mean()
    df["atl"] = df["tss"].ewm(span=7, adjust=False).mean()
    df["tsb"] = df["ctl"] - df["atl"]

    if "weight" in df.columns:
        weight = pd.to_numeric(df["weight"], errors="coerce").fillna(WEIGHT_KG)
    else:
        weight = pd.Series([WEIGHT_KG] * len(df))
    df["w_per_kg"] = pd.to_numeric(df["power_avg"], errors="coerce") / weight

    hr  = pd.to_numeric(df["hr_avg"], errors="coerce")
    pwr = pd.to_numeric(df["power_avg"], errors="coerce")
    df["efficiency"] = np.where(hr > 0, pwr / hr, np.nan)

    if_s = pd.to_numeric(df["if_score"], errors="coerce")
    dur  = pd.to_numeric(df["duration_h"], errors="coerce")
    df["ftp_stimulus"] = (if_s ** 2) * dur * 100

    conditions = [
        df["ctl"] < 30,
        df["tsb"] < -30,
        df["tsb"] < -10,
        df["tsb"] < 0,
        df["tsb"] < 10,
        df["tsb"] < 25,
    ]
    choices = [
        "Undertrained", "Overreached", "Deep Block",
        "Build Phase", "Neutral", "Fresh"
    ]
    df["fatigue_state"] = np.select(conditions, choices, default="Peak/Detrain Risk")
    return df


@st.cache_data(ttl=3600)
def load_wellness():
    p = "data/wellness_data.csv"
    if Path(p).exists():
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


df_all  = load_data()
wellness = load_wellness()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚴 Cycling Performance")
    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("### Filters")
    date_range = st.selectbox(
        "Time Range",
        ["Last 30 days", "Last 90 days", "Last 6 months",
         "Last 12 months", "All time"],
        index=2
    )

    available_types = ["All"] + sorted(
        df_all["training_type"].dropna().unique().tolist()
    )
    type_filter = st.multiselect(
        "Training Types", available_types, default=["All"]
    )

    st.markdown("---")
    st.markdown(f"**Last updated:** {datetime.now().strftime('%d %b %Y %H:%M')}")
    st.markdown(f"**Total sessions:** {len(df_all)}")
    st.markdown(
        f"**Date range:** {df_all['date'].min().date()} → {df_all['date'].max().date()}"
    )

# ── Filter ────────────────────────────────────────────────────────────────────
days_map = {
    "Last 30 days": 30, "Last 90 days": 90,
    "Last 6 months": 180, "Last 12 months": 365, "All time": 9999
}
cutoff = datetime.now() - timedelta(days=days_map[date_range])
df = df_all[df_all["date"] >= cutoff].copy()
df_main_all = df_all[df_all["training_type"].isin(MAIN_TYPES)].copy()
df_main     = df[df["training_type"].isin(MAIN_TYPES)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🚴 Cycling Performance Dashboard")
st.markdown(f"*{date_range} · {len(df)} sessions · Catalonia*")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📅 This Week vs Last Week")

now          = pd.Timestamp.now()
week_start   = now - timedelta(days=now.weekday())
last_week_s  = week_start - timedelta(days=7)

this_week = df_all[df_all["date"] >= week_start]
last_week = df_all[(df_all["date"] >= last_week_s) & (df_all["date"] < week_start)]

def safe_mean(series):
    v = pd.to_numeric(series, errors="coerce").dropna()
    return v.mean() if len(v) > 0 else 0

tw_tss  = pd.to_numeric(this_week["tss"], errors="coerce").sum()
lw_tss  = pd.to_numeric(last_week["tss"], errors="coerce").sum()
tw_sess = len(this_week)
lw_sess = len(last_week)
tw_if   = safe_mean(this_week["if_score"])
lw_if   = safe_mean(last_week["if_score"])
tw_pwr  = safe_mean(this_week["power_avg"])
lw_pwr  = safe_mean(last_week["power_avg"])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Weekly TSS", f"{tw_tss:.0f}",
              delta=f"{tw_tss - lw_tss:+.0f} vs last week")
with col2:
    st.metric("Sessions", f"{tw_sess}",
              delta=f"{tw_sess - lw_sess:+d} vs last week")
with col3:
    st.metric("Avg IF", f"{tw_if:.3f}" if tw_if > 0 else "—",
              delta=f"{tw_if - lw_if:+.3f}" if tw_if > 0 and lw_if > 0 else None)
with col4:
    st.metric("Avg Power", f"{tw_pwr:.0f}W" if tw_pwr > 0 else "—",
              delta=f"{tw_pwr - lw_pwr:+.0f}W" if tw_pwr > 0 and lw_pwr > 0 else None)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TODAY'S STATUS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Today's Status")

latest      = df_all.iloc[-1]
prev        = df_all.iloc[-8] if len(df_all) >= 8 else df_all.iloc[0]
ctl_now     = latest["ctl"]
atl_now     = latest["atl"]
tsb_now     = latest["tsb"]
state       = latest["fatigue_state"]
state_color = STATE_COLORS.get(state, C["accent"])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("CTL (Fitness)", f"{ctl_now:.1f}",
              delta=f"{ctl_now - prev['ctl']:+.1f} vs 7d ago")
with col2:
    st.metric("ATL (Fatigue)", f"{atl_now:.1f}",
              delta=f"{atl_now - prev['atl']:+.1f} vs 7d ago")
with col3:
    st.metric("TSB (Form)", f"{tsb_now:+.1f}",
              delta=f"{tsb_now - prev['tsb']:+.1f} vs 7d ago")
with col4:
    if "eftp" in df_all.columns:
        eftp_s = pd.to_numeric(df_all["eftp"], errors="coerce").dropna()
        if len(eftp_s) > 0:
            ev = eftp_s.iloc[-1]
            ep = eftp_s.iloc[-2] if len(eftp_s) > 1 else ev
            st.metric("eFTP", f"{ev:.0f}W", delta=f"{ev - ep:+.0f}W")
        else:
            st.metric("eFTP", "—")
    else:
        st.metric("eFTP", "—")
with col5:
    wkg_s   = pd.to_numeric(df_all["w_per_kg"], errors="coerce").dropna()
    wkg_val = wkg_s.tail(10).mean() if len(wkg_s) >= 10 else wkg_s.mean()
    st.metric("W/kg (10-session avg)", f"{wkg_val:.2f}")

st.markdown(
    f'<div class="state-box" style="background:{state_color}22; '
    f'border: 2px solid {state_color}; color:{state_color};">'
    f'Current State: {state}</div>',
    unsafe_allow_html=True
)

state_guide = {
    "Undertrained":       "⚠️ Build baseline volume — CTL too low",
    "Overreached":        "🚨 Rest immediately — TSB below -30",
    "Deep Block":         "💪 Hard training block — monitor recovery closely",
    "Build Phase":        "✅ Most productive training zone — keep pushing",
    "Neutral":            "⚖️ Balanced load — maintain or start taper",
    "Fresh":              "🟢 Race-ready window — quality sessions now",
    "Peak/Detrain Risk":  "⚡ Peak form — race or start next block",
}
st.info(state_guide.get(state, ""))

if "icu_ctl" in df_all.columns and "icu_atl" in df_all.columns:
    _icu_f = pd.to_numeric(df_all["icu_ctl"], errors="coerce").dropna()
    _icu_a = pd.to_numeric(df_all["icu_atl"], errors="coerce").dropna()
    if len(_icu_f) > 0 and len(_icu_a) > 0:
        st.caption(
            f"Intervals.icu (last synced activity): "
            f"Fitness {_icu_f.iloc[-1]:.0f} · Fatigue {_icu_a.iloc[-1]:.0f} · "
            f"Form {_icu_f.iloc[-1] - _icu_a.iloc[-1]:+.0f}"
        )
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FTP STIMULUS — THE KEY CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 FTP Development Analysis")
st.caption("Is your training mix actually driving FTP up?")

col1, col2 = st.columns(2)

with col1:
    # FTP Stimulus Score by training type — filtered period
    if len(df_main) > 0:
        stim = (
            df_main.groupby("training_type")["ftp_stimulus"]
            .mean()
            .reset_index()
            .sort_values("ftp_stimulus", ascending=True)
        )
        stim["is_driver"] = stim["training_type"].isin(FTP_DRIVERS)
        bar_colors = [C["green"] if d else C["muted"] for d in stim["is_driver"]]

        fig_stim = go.Figure()
        fig_stim.add_trace(go.Bar(
            x=stim["ftp_stimulus"],
            y=stim["training_type"],
            orientation="h",
            marker_color=bar_colors,
            opacity=0.85,
            text=[f"{v:.1f}" for v in stim["ftp_stimulus"]],
            textposition="outside",
        ))
        fig_stim.update_layout(
            title=f"FTP Stimulus Score — {date_range}<br>"
                  f"<sup>IF² × Duration × 100 | Green = FTP driver types</sup>",
            height=420, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_stim, use_container_width=True)
    else:
        st.info("No labelled training type data in this period.")

with col2:
    # FTP driver % of total sessions — pie chart
    if len(df_main) > 0:
        driver_count = df_main["training_type"].isin(FTP_DRIVERS).sum()
        base_count   = len(df_main) - driver_count
        driver_tss   = df_main.loc[df_main["training_type"].isin(FTP_DRIVERS), "tss"]
        base_tss     = df_main.loc[~df_main["training_type"].isin(FTP_DRIVERS), "tss"]

        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=["FTP Drivers (FTP/SST/TEMPO/VO2)", "Base Volume (END/Z2/FATMAX)"],
            values=[
                pd.to_numeric(driver_tss, errors="coerce").sum(),
                pd.to_numeric(base_tss, errors="coerce").sum()
            ],
            marker_colors=[C["green"], C["muted"]],
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            title="TSS Split — Quality vs Base Volume",
            height=420,
            paper_bgcolor=C["bg"],
            font=dict(color=C["text"]),
            legend=dict(bgcolor=C["panel"]),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Insight text
        total_main = len(df_main)
        driver_pct = driver_count / total_main * 100 if total_main > 0 else 0
        if driver_pct < 25:
            insight_color = C["orange"]
            insight = (
                f"⚠️ Only {driver_pct:.0f}% of sessions are FTP drivers. "
                "You need more FTP/SST/TEMPO work to push threshold up."
            )
        elif driver_pct < 40:
            insight_color = C["yellow"]
            insight = (
                f"✅ {driver_pct:.0f}% FTP driver sessions — good balance. "
                "2 quality sessions per week is the target."
            )
        else:
            insight_color = C["green"]
            insight = (
                f"🔥 {driver_pct:.0f}% FTP driver sessions — high intensity block. "
                "Monitor recovery — don't skip base volume."
            )
        st.markdown(
            f'<div style="background:{insight_color}22; border:1px solid {insight_color}; '
            f'border-radius:8px; padding:12px; margin-top:8px; color:{C["text"]};">'
            f'{insight}</div>',
            unsafe_allow_html=True
        )

# FTP Stimulus trend over time
if len(df_main) > 0:
    weekly_stim = (
        df_main.groupby(pd.Grouper(key="date", freq="W"))
        .agg(
            total_stimulus=("ftp_stimulus", "sum"),
            driver_stimulus=("ftp_stimulus", lambda x: x[
                df_main.loc[x.index, "training_type"].isin(FTP_DRIVERS)
            ].sum()),
            sessions=("tss", "count"),
        )
        .reset_index()
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=weekly_stim["date"],
        y=weekly_stim["total_stimulus"],
        name="Total FTP Stimulus",
        marker_color=C["muted"],
        opacity=0.5,
    ))
    fig_trend.add_trace(go.Bar(
        x=weekly_stim["date"],
        y=weekly_stim["driver_stimulus"],
        name="FTP Driver Sessions Only",
        marker_color=C["green"],
        opacity=0.85,
    ))
    fig_trend.add_trace(go.Scatter(
        x=weekly_stim["date"],
        y=weekly_stim["total_stimulus"].rolling(4).mean(),
        name="4-week trend",
        line=dict(color=C["accent"], width=2.5),
        mode="lines",
    ))
    fig_trend.update_layout(
        barmode="overlay",
        title="Weekly FTP Stimulus Trend — Are quality sessions increasing?",
        height=320, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PMC CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📈 Performance Management Chart")

fig_pmc = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.4, 0.35, 0.25],
    subplot_titles=[
        "CTL vs ATL (Fitness vs Fatigue)",
        "TSB — Form / Freshness",
        "Weekly TSS"
    ]
)

fig_pmc.add_trace(go.Scatter(
    x=df["date"], y=df["ctl"], name="CTL Fitness",
    line=dict(color=C["green"], width=2.5)
), row=1, col=1)

fig_pmc.add_trace(go.Scatter(
    x=df["date"], y=df["atl"], name="ATL Fatigue",
    line=dict(color=C["orange"], width=2.5),
    fill="tonexty", fillcolor="rgba(240,136,62,0.1)"
), row=1, col=1)

for state_name, color in STATE_COLORS.items():
    mask = df["fatigue_state"] == state_name
    if mask.sum() > 0:
        fig_pmc.add_trace(go.Scatter(
            x=df.loc[mask, "date"],
            y=df.loc[mask, "tsb"],
            mode="markers",
            name=state_name,
            marker=dict(color=color, size=5, opacity=0.7),
        ), row=2, col=1)

fig_pmc.add_hline(y=-30, line_dash="dash", line_color=C["red"],
                   annotation_text="-30 Overreach", row=2, col=1)
fig_pmc.add_hline(y=25, line_dash="dash", line_color=C["purple"],
                   annotation_text="+25 Peak", row=2, col=1)
fig_pmc.add_hline(y=0, line_dash="dot", line_color=C["muted"], row=2, col=1)

weekly = df.resample("W", on="date")["tss"].sum().reset_index()
bar_colors_pmc = [
    C["red"] if t >= 700 else C["green"] if t >= 400 else C["yellow"]
    for t in weekly["tss"]
]
fig_pmc.add_trace(go.Bar(
    x=weekly["date"], y=weekly["tss"],
    marker_color=bar_colors_pmc, name="Weekly TSS", opacity=0.75
), row=3, col=1)

fig_pmc.update_layout(
    height=750, showlegend=True,
    paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
    font=dict(color=C["text"]),
    legend=dict(bgcolor=C["panel"], bordercolor=C["grid"]),
)
for i in range(1, 4):
    fig_pmc.update_xaxes(gridcolor=C["grid"], row=i, col=1)
    fig_pmc.update_yaxes(gridcolor=C["grid"], row=i, col=1)

st.plotly_chart(fig_pmc, use_container_width=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TIME IN POWER ZONES — from Intervals.icu per-second data
# ══════════════════════════════════════════════════════════════════════════════
ZONE_COLS   = [f"z{i}_secs" for i in range(1, 8)]
ZONE_NAMES  = ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold",
               "Z5 VO2max", "Z6 Anaerobic", "Z7 Neuromuscular"]
ZONE_COLORS = [C["muted"], C["accent"], C["yellow"], C["orange"],
               C["red"], C["purple"], "#e6edf3"]

zone_cols_present = [c for c in ZONE_COLS if c in df.columns]
if len(zone_cols_present) >= 5:
    zone_df = df[["date"] + zone_cols_present].copy()
    for zc in zone_cols_present:
        zone_df[zc] = pd.to_numeric(zone_df[zc], errors="coerce").fillna(0)

    if zone_df[zone_cols_present].sum().sum() > 0:
        st.markdown("## 🌈 Time in Power Zones")
        st.caption(f"Source: Intervals.icu zone data · {date_range}")

        col1, col2 = st.columns(2)

        with col1:
            totals_h = zone_df[zone_cols_present].sum() / 3600.0
            fig_tz = go.Figure()
            fig_tz.add_trace(go.Bar(
                x=totals_h.values,
                y=[ZONE_NAMES[ZONE_COLS.index(c)] for c in zone_cols_present],
                orientation="h",
                marker_color=[ZONE_COLORS[ZONE_COLS.index(c)]
                              for c in zone_cols_present],
                text=[f"{v:.1f}h" for v in totals_h.values],
                textposition="outside",
            ))
            fig_tz.update_layout(title="Total Hours per Zone",
                                 height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_tz, use_container_width=True)

        with col2:
            weekly_z = (zone_df.resample("W", on="date")[zone_cols_present]
                        .sum() / 3600.0)
            fig_wz = go.Figure()
            for zc in zone_cols_present:
                fig_wz.add_trace(go.Bar(
                    x=weekly_z.index, y=weekly_z[zc],
                    name=ZONE_NAMES[ZONE_COLS.index(zc)],
                    marker_color=ZONE_COLORS[ZONE_COLS.index(zc)],
                ))
            fig_wz.update_layout(barmode="stack",
                                 title="Weekly Hours per Zone",
                                 height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_wz, use_container_width=True)

        total_secs = zone_df[zone_cols_present].sum().sum()
        low_cols = [c for c in ["z1_secs", "z2_secs"] if c in zone_cols_present]
        hi_cols  = [c for c in ["z4_secs", "z5_secs", "z6_secs", "z7_secs"]
                    if c in zone_cols_present]
        low_pct = zone_df[low_cols].sum().sum() / total_secs * 100
        hi_pct  = zone_df[hi_cols].sum().sum() / total_secs * 100
        st.markdown(
            f'<div style="background:{C["accent"]}22; border:1px solid '
            f'{C["accent"]}; border-radius:8px; padding:12px; '
            f'color:{C["text"]};">'
            f'⚖️ Intensity distribution: <b>{low_pct:.0f}%</b> low (Z1–Z2) · '
            f'<b>{100 - low_pct - hi_pct:.0f}%</b> tempo (Z3) · '
            f'<b>{hi_pct:.0f}%</b> high (Z4+). '
            f'Pyramidal reference for a climber: ~75–85% low.</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# POWER & EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ⚡ Power & Efficiency")

col1, col2 = st.columns(2)

with col1:
    monthly = df.resample("ME", on="date").agg(
        avg_wkg=("w_per_kg", "mean"),
        max_wkg=("w_per_kg", "max"),
    ).reset_index()

    fig_wkg = go.Figure()
    fig_wkg.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["avg_wkg"],
        mode="lines+markers", name="Avg W/kg",
        line=dict(color=C["purple"], width=2.5),
        marker=dict(size=6)
    ))
    fig_wkg.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["max_wkg"],
        mode="lines", name="Max W/kg",
        line=dict(color=C["accent"], width=1.5, dash="dash"),
        opacity=0.6
    ))
    fig_wkg.add_hline(
        y=WKG_CURRENT, line_dash="dot", line_color=C["yellow"],
        annotation_text=f"Current FTP ({WKG_CURRENT:.2f} W/kg = {FTP_CURRENT}W)"
    )
    fig_wkg.add_hline(
        y=WKG_PRE, line_dash="dot", line_color=C["purple"],
        annotation_text=f"Pre-accident ({WKG_PRE:.2f} W/kg = {FTP_PRE}W)",
        opacity=0.4
    )
    fig_wkg.update_layout(
        title="Monthly W/kg Trend", height=350, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_wkg, use_container_width=True)

with col2:
    monthly_eff = df.resample("ME", on="date").agg(
        avg_eff=("efficiency", "mean")
    ).reset_index().dropna()

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=monthly_eff["date"], y=monthly_eff["avg_eff"],
        mode="lines+markers", name="W/BPM",
        line=dict(color=C["orange"], width=2.5),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(240,136,62,0.1)"
    ))
    fig_eff.update_layout(
        title="Cardiac Efficiency (W per BPM) — Higher = Better",
        height=350, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_eff, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# POWER CURVE — best efforts by duration bucket
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔋 Power Curve — Best Efforts by Duration")
st.caption(
    "Best and average power per session-duration bucket. Built from session "
    "averages (one value per ride), not second-by-second peaks."
)

DUR_BINS   = [0, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 24.0]
DUR_LABELS = ["<45min", "45–75min", "1h15–1h45", "1h45–2h30",
              "2h30–3h30", "3h30–5h", "5h+"]


def build_power_curve(frame):
    d = frame.copy()
    d["power_avg"]  = pd.to_numeric(d["power_avg"], errors="coerce")
    d["duration_h"] = pd.to_numeric(d["duration_h"], errors="coerce")
    d = d.dropna(subset=["power_avg", "duration_h"])
    d = d[(d["power_avg"] > 50) & (d["duration_h"] > 0.1)]
    if len(d) == 0:
        return pd.DataFrame()
    d["bucket"] = pd.cut(d["duration_h"], bins=DUR_BINS, labels=DUR_LABELS)
    out = (
        d.groupby("bucket", observed=False)
        .agg(
            best_power=("power_avg", "max"),
            avg_power=("power_avg", "mean"),
            best_wkg=("w_per_kg", "max"),
            sessions=("power_avg", "count"),
        )
        .reset_index()
    )
    return out[out["sessions"] > 0]


curve_all    = build_power_curve(df_all)
curve_period = build_power_curve(df)

if len(curve_all) > 0:
    col1, col2 = st.columns(2)

    with col1:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=curve_all["bucket"].astype(str), y=curve_all["best_power"],
            mode="lines+markers", name="All-time best",
            line=dict(color=C["purple"], width=2.5), marker=dict(size=8),
        ))
        fig_curve.add_trace(go.Scatter(
            x=curve_all["bucket"].astype(str), y=curve_all["avg_power"],
            mode="lines+markers", name="All-time average",
            line=dict(color=C["muted"], width=1.5, dash="dash"),
            marker=dict(size=5),
        ))
        if len(curve_period) > 0 and date_range != "All time":
            fig_curve.add_trace(go.Scatter(
                x=curve_period["bucket"].astype(str),
                y=curve_period["best_power"],
                mode="lines+markers", name=f"Best — {date_range}",
                line=dict(color=C["green"], width=2.5), marker=dict(size=8),
            ))
        fig_curve.add_hline(y=FTP_CURRENT, line_dash="dot",
                            line_color=C["yellow"],
                            annotation_text=f"FTP {FTP_CURRENT}W")
        fig_curve.update_layout(
            title="Best Avg Power per Duration",
            height=380, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with col2:
        fig_wkg_curve = go.Figure()
        fig_wkg_curve.add_trace(go.Scatter(
            x=curve_all["bucket"].astype(str), y=curve_all["best_wkg"],
            mode="lines+markers", name="All-time best W/kg",
            line=dict(color=C["purple"], width=2.5), marker=dict(size=8),
        ))
        if len(curve_period) > 0 and date_range != "All time":
            fig_wkg_curve.add_trace(go.Scatter(
                x=curve_period["bucket"].astype(str),
                y=curve_period["best_wkg"],
                mode="lines+markers", name=f"Best W/kg — {date_range}",
                line=dict(color=C["green"], width=2.5), marker=dict(size=8),
            ))
        fig_wkg_curve.add_hline(y=WKG_CURRENT, line_dash="dot",
                                line_color=C["yellow"],
                                annotation_text=f"{WKG_CURRENT:.2f} W/kg (current FTP)")
        fig_wkg_curve.update_layout(
            title="Best W/kg per Duration (climber's view)",
            height=380, **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_wkg_curve, use_container_width=True)

    # Gap analysis: period best vs all-time best
    if len(curve_period) > 0 and date_range != "All time":
        merged = curve_all.merge(
            curve_period, on="bucket", suffixes=("_all", "_now")
        )
        if len(merged) > 0:
            merged["gap_pct"] = (
                (merged["best_power_now"] - merged["best_power_all"])
                / merged["best_power_all"] * 100
            )
            worst = merged.loc[merged["gap_pct"].idxmin()]
            st.markdown(
                f'<div style="background:{C["accent"]}22; border:1px solid '
                f'{C["accent"]}; border-radius:8px; padding:12px; '
                f'color:{C["text"]};">'
                f'📏 Biggest gap to all-time best: <b>{worst["bucket"]}</b> '
                f'rides ({worst["gap_pct"]:+.1f}% — '
                f'{worst["best_power_now"]:.0f}W now vs '
                f'{worst["best_power_all"]:.0f}W all-time).</div>',
                unsafe_allow_html=True
            )
else:
    st.info("Not enough power data to build the curve.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FTP PROGRESSION — month by month
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📶 FTP Progression — Month by Month")

_eftp_num = (
    pd.to_numeric(df_all["eftp"], errors="coerce")
    if "eftp" in df_all.columns else pd.Series(dtype=float)
)
has_eftp = _eftp_num.notna().sum() > 3

if has_eftp:
    src = df_all.copy()
    src["ftp_metric"] = _eftp_num
    ftp_label = "eFTP (Intervals.icu)"
else:
    src = df_all.copy()
    src["power_avg"]  = pd.to_numeric(src["power_avg"], errors="coerce")
    src["if_score"]   = pd.to_numeric(src["if_score"], errors="coerce")
    src["duration_h"] = pd.to_numeric(src["duration_h"], errors="coerce")
    src["ftp_metric"] = np.where(
        (src["if_score"] >= 0.78) & (src["duration_h"] >= 0.75),
        src["power_avg"], np.nan
    )
    ftp_label = "Threshold proxy — best avg power in hard rides (IF ≥ 0.78, ≥ 45 min)"

monthly_ftp = (
    src.dropna(subset=["ftp_metric"])
    .resample("ME", on="date")["ftp_metric"]
    .max()
    .reset_index()
    .dropna()
)

if len(monthly_ftp) > 1:
    monthly_ftp["trend3"] = monthly_ftp["ftp_metric"].rolling(3, min_periods=1).mean()

    fig_ftp = go.Figure()
    fig_ftp.add_trace(go.Bar(
        x=monthly_ftp["date"], y=monthly_ftp["ftp_metric"],
        name=ftp_label, marker_color=C["accent"], opacity=0.55,
    ))
    fig_ftp.add_trace(go.Scatter(
        x=monthly_ftp["date"], y=monthly_ftp["trend3"],
        name="3-month trend", mode="lines",
        line=dict(color=C["green"], width=3),
    ))
    fig_ftp.add_hline(y=FTP_CURRENT, line_dash="dot", line_color=C["yellow"],
                      annotation_text=f"Current FTP {FTP_CURRENT}W")
    fig_ftp.add_hline(y=FTP_PRE, line_dash="dot", line_color=C["purple"],
                      annotation_text=f"Pre-accident {FTP_PRE}W", opacity=0.4)
    try:
        fig_ftp.add_vline(x="2025-06-15", line_dash="dash",
                          line_color=C["red"],
                          annotation_text="Surgery", opacity=0.6)
    except Exception:
        pass
    fig_ftp.update_layout(
        title=f"Monthly FTP Progression — {ftp_label}",
        height=400, barmode="overlay", **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_ftp, use_container_width=True)

    last6 = monthly_ftp.tail(6)
    if len(last6) >= 2:
        delta6 = last6["ftp_metric"].iloc[-1] - last6["ftp_metric"].iloc[0]
        d_color = C["green"] if delta6 >= 0 else C["red"]
        st.markdown(
            f'<div style="background:{d_color}22; border:1px solid {d_color}; '
            f'border-radius:8px; padding:12px; color:{C["text"]};">'
            f'{"📈" if delta6 >= 0 else "📉"} Last 6 months: '
            f'<b>{delta6:+.0f}W</b> '
            f'({last6["ftp_metric"].iloc[0]:.0f}W → '
            f'{last6["ftp_metric"].iloc[-1]:.0f}W).</div>',
            unsafe_allow_html=True
        )
else:
    st.info("Not enough data to build the FTP progression chart.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ELEVATION — climbing volume
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ⛰️ Climbing Volume")

if "elevation" in df.columns:
    elev_series = pd.to_numeric(df["elevation"], errors="coerce").fillna(0)
    total_elev  = elev_series.sum()
    dur_sum     = pd.to_numeric(df["duration_h"], errors="coerce").sum()
    climb_rate  = total_elev / dur_sum if dur_sum > 0 else 0

    weekly_elev = (
        df.assign(elev=elev_series)
        .resample("W", on="date")["elev"]
        .sum()
        .reset_index()
    )
    weeks_n    = max(len(weekly_elev), 1)
    weekly_avg = total_elev / weeks_n
    best_week  = weekly_elev["elev"].max() if len(weekly_elev) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Elevation", f"{total_elev:,.0f} m")
    with col2:
        st.metric("Weekly Average", f"{weekly_avg:,.0f} m")
    with col3:
        st.metric("Best Week", f"{best_week:,.0f} m")
    with col4:
        st.metric("Climbing Rate", f"{climb_rate:.0f} m/h")

    elev_colors = [
        C["purple"] if v >= 4000 else
        C["green"]  if v >= 2500 else
        C["yellow"] if v >= 1000 else C["muted"]
        for v in weekly_elev["elev"]
    ]
    fig_elev = go.Figure()
    fig_elev.add_trace(go.Bar(
        x=weekly_elev["date"], y=weekly_elev["elev"],
        marker_color=elev_colors, name="Weekly elevation", opacity=0.8,
    ))
    fig_elev.add_trace(go.Scatter(
        x=weekly_elev["date"],
        y=weekly_elev["elev"].rolling(4, min_periods=1).mean(),
        mode="lines", name="4-week trend",
        line=dict(color=C["accent"], width=2.5),
    ))
    fig_elev.update_layout(
        title="Weekly Elevation Gain — Purple ≥4,000m · Green ≥2,500m · Yellow ≥1,000m",
        height=350, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_elev, use_container_width=True)
else:
    st.info("No elevation column found in the dataset.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# THIS BLOCK vs SAME PERIOD LAST YEAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔄 This Block vs Same Period Last Year")

yoy_days = days_map[date_range] if date_range != "All time" else 365
yoy_label = date_range if date_range != "All time" else "Last 12 months"
st.caption(f"Comparing {yoy_label.lower()} vs the same window one year earlier.")

cur_start  = now - timedelta(days=yoy_days)
prev_start = cur_start - timedelta(days=365)
prev_end   = now - timedelta(days=365)

cur_blk  = df_all[df_all["date"] >= cur_start].copy()
prev_blk = df_all[
    (df_all["date"] >= prev_start) & (df_all["date"] < prev_end)
].copy()


def block_stats(frame):
    if len(frame) == 0:
        return dict(sessions=0, tss=0, hours=0, elev=0, avg_if=0,
                    avg_wkg=0, ctl_gain=0)
    tss   = pd.to_numeric(frame["tss"], errors="coerce").sum()
    hours = pd.to_numeric(frame["duration_h"], errors="coerce").sum()
    elev  = (pd.to_numeric(frame["elevation"], errors="coerce").sum()
             if "elevation" in frame.columns else 0)
    return dict(
        sessions=len(frame),
        tss=tss,
        hours=hours,
        elev=elev,
        avg_if=safe_mean(frame["if_score"]),
        avg_wkg=safe_mean(frame["w_per_kg"]),
        ctl_gain=float(frame["ctl"].iloc[-1] - frame["ctl"].iloc[0]),
    )


cs = block_stats(cur_blk)
ps = block_stats(prev_blk)

if ps["sessions"] == 0:
    st.info("No data for the same period last year.")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sessions", f"{cs['sessions']}",
                  delta=f"{cs['sessions'] - ps['sessions']:+d} vs last year")
    with col2:
        st.metric("Total TSS", f"{cs['tss']:.0f}",
                  delta=f"{cs['tss'] - ps['tss']:+.0f} vs last year")
    with col3:
        st.metric("Hours", f"{cs['hours']:.0f}h",
                  delta=f"{cs['hours'] - ps['hours']:+.0f}h vs last year")
    with col4:
        st.metric("Elevation", f"{cs['elev']:,.0f} m",
                  delta=f"{cs['elev'] - ps['elev']:+,.0f} m vs last year")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Avg IF", f"{cs['avg_if']:.3f}" if cs['avg_if'] > 0 else "—",
                  delta=(f"{cs['avg_if'] - ps['avg_if']:+.3f}"
                         if cs['avg_if'] > 0 and ps['avg_if'] > 0 else None))
    with col6:
        st.metric("Avg W/kg", f"{cs['avg_wkg']:.2f}" if cs['avg_wkg'] > 0 else "—",
                  delta=(f"{cs['avg_wkg'] - ps['avg_wkg']:+.2f}"
                         if cs['avg_wkg'] > 0 and ps['avg_wkg'] > 0 else None))
    with col7:
        st.metric("CTL Change", f"{cs['ctl_gain']:+.1f}",
                  delta=f"{cs['ctl_gain'] - ps['ctl_gain']:+.1f} vs last year")

    def weekly_aligned(frame, start):
        if len(frame) == 0:
            return pd.DataFrame(columns=["week", "tss"])
        w = frame.resample("W", on="date")["tss"].sum().reset_index()
        w["week"] = ((w["date"] - pd.Timestamp(start)).dt.days // 7) + 1
        return w[w["week"] >= 1]

    cur_w  = weekly_aligned(cur_blk, cur_start)
    prev_w = weekly_aligned(prev_blk, prev_start)

    fig_yoy = go.Figure()
    fig_yoy.add_trace(go.Bar(
        x=prev_w["week"], y=prev_w["tss"],
        name="Last year", marker_color=C["muted"], opacity=0.5,
    ))
    fig_yoy.add_trace(go.Bar(
        x=cur_w["week"], y=cur_w["tss"],
        name="This block", marker_color=C["green"], opacity=0.85,
    ))
    fig_yoy.update_layout(
        barmode="overlay",
        title=f"Weekly TSS — {yoy_label} vs Same Window Last Year",
        height=330,
        **PLOTLY_LAYOUT
    )
    fig_yoy.update_xaxes(title_text="Week of block")
    st.plotly_chart(fig_yoy, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING TYPE ANALYSIS — all time
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏋️ Training Type Analysis")
st.caption("Showing all-time data regardless of time filter")

col1, col2 = st.columns(2)

with col1:
    type_counts = df_main_all["training_type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]

    fig_types = px.bar(
        type_counts, x="count", y="type", orientation="h",
        color="type", title="Sessions per Training Type (All Time)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_types.update_layout(height=400, showlegend=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_types, use_container_width=True)

with col2:
    if "if_score" in df_main_all.columns:
        type_if = (
            df_main_all.groupby("training_type")["if_score"]
            .mean()
            .reset_index()
            .sort_values("if_score", ascending=True)
        )
        fig_if = px.bar(
            type_if, x="if_score", y="training_type", orientation="h",
            color="if_score", title="Avg Intensity Factor by Type (All Time)",
            color_continuous_scale=[[0, C["accent"]], [0.5, C["yellow"]], [1, C["red"]]]
        )
        fig_if.add_vline(x=0.75, line_dash="dash", line_color=C["orange"],
                          annotation_text="SST (0.75)")
        fig_if.add_vline(x=0.85, line_dash="dash", line_color=C["red"],
                          annotation_text="FTP (0.85)")
        fig_if.update_layout(height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_if, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RECENT SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📅 Recent Sessions")

display_cols = [
    c for c in [
        "date", "training_type", "tss", "if_score",
        "power_avg", "duration_h", "hr_avg",
        "w_per_kg", "elevation", "fatigue_state"
    ]
    if c in df_all.columns
]

recent = df_all[display_cols].tail(20).sort_values("date", ascending=False).copy()
recent["date"] = recent["date"].dt.strftime("%d %b %Y")

for col in ["tss", "if_score", "power_avg", "duration_h", "hr_avg", "w_per_kg", "elevation"]:
    if col in recent.columns:
        recent[col] = pd.to_numeric(recent[col], errors="coerce").round(2)

if "training_type" in recent.columns:
    recent["training_type"] = recent["training_type"].fillna("—")

recent = recent.rename(columns={
    "date": "Date", "training_type": "Type", "tss": "TSS",
    "if_score": "IF", "power_avg": "Power (W)", "duration_h": "Hours",
    "hr_avg": "Avg HR", "w_per_kg": "W/kg", "elevation": "Elev (m)",
    "fatigue_state": "State"
})

st.dataframe(
    recent,
    use_container_width=True,
    height=450,
    column_config={
        "TSS": st.column_config.ProgressColumn(
            "TSS", min_value=0, max_value=300, format="%d"
        ),
        "IF": st.column_config.NumberColumn("IF", format="%.3f"),
        "W/kg": st.column_config.NumberColumn("W/kg", format="%.2f"),
    }
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# WELLNESS
# ══════════════════════════════════════════════════════════════════════════════
has_hrv = (
    not wellness.empty
    and "hrv" in wellness.columns
    and wellness["hrv"].notna().sum() > 5
)
has_rhr = (
    not wellness.empty
    and "resting_hr" in wellness.columns
    and wellness["resting_hr"].notna().sum() > 5
)

if has_hrv or has_rhr:
    st.markdown("## 💚 Wellness & HRV")
    wellness_recent = wellness[wellness["date"] >= pd.Timestamp(cutoff)]
    col1, col2 = st.columns(2)

    with col1:
        if has_hrv:
            fig_hrv = go.Figure()
            fig_hrv.add_trace(go.Scatter(
                x=wellness_recent["date"],
                y=wellness_recent["hrv"],
                mode="lines+markers",
                line=dict(color=C["green"], width=2),
                name="HRV",
                fill="tozeroy",
                fillcolor="rgba(63,185,80,0.1)"
            ))
            fig_hrv.update_layout(
                title="Heart Rate Variability (HRV)",
                height=300, **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig_hrv, use_container_width=True)

    with col2:
        if has_rhr:
            fig_rhr = go.Figure()
            fig_rhr.add_trace(go.Scatter(
                x=wellness_recent["date"],
                y=wellness_recent["resting_hr"],
                mode="lines+markers",
                line=dict(color=C["orange"], width=2),
                name="Resting HR"
            ))
            fig_rhr.update_layout(
                title="Resting Heart Rate",
                height=300, **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig_rhr, use_container_width=True)
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# NEXT WEEK RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🗓️ Next Week Recommendation")

tsb_now = float(df_all.iloc[-1]["tsb"])
ctl_now = float(df_all.iloc[-1]["ctl"])
atl_now = float(df_all.iloc[-1]["atl"])

if tsb_now < -30:
    rc = C["red"]
    rt = "🚨 Rest or Very Easy Week"
    rx = (f"TSB = {tsb_now:.1f} — Overreached. "
          "Max 3 easy Z2 sessions. No quality work until TSB > -20.")
elif tsb_now < -10:
    rc = C["orange"]
    rt = "💪 Continue Hard Block"
    rx = (f"TSB = {tsb_now:.1f} — Deep build phase. "
          "2 quality sessions: Wednesday FTP intervals + Saturday PIRAMIDAL.")
elif tsb_now < 5:
    rc = C["yellow"]
    rt = "✅ Standard Build Week"
    rx = (f"TSB = {tsb_now:.1f} — Build phase. "
          "Wednesday FTP/SST + Saturday PIRAMIDAL with TEMPO blocks.")
elif tsb_now < 20:
    rc = C["green"]
    rt = "🟢 Push Hard This Week"
    rx = (f"TSB = {tsb_now:.1f} — Fresh and ready. "
          "4x10min FTP Wednesday + long PIRAMIDAL Saturday.")
else:
    rc = C["purple"]
    rt = "⚡ Peak Form — Race or FTP Test"
    rx = (f"TSB = {tsb_now:.1f} — Peak form. "
          "Do a 20-min FTP test or your hardest session of the block.")

st.markdown(
    f'<div style="background:{rc}22; border: 2px solid {rc}; '
    f'border-radius: 10px; padding: 20px; margin: 10px 0;">'
    f'<h3 style="color:{rc}; margin:0 0 10px 0;">{rt}</h3>'
    f'<p style="color:#c9d1d9; margin:0; font-size:1.05rem;">{rx}</p>'
    f'<p style="color:#8b949e; margin:10px 0 0 0; font-size:0.85rem;">'
    f'CTL={ctl_now:.1f} · ATL={atl_now:.1f} · TSB={tsb_now:+.1f}</p>'
    f'</div>',
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    "*Run `python src/intervals_api.py` to fetch latest data · "
    "Then click Refresh Data*"
)
