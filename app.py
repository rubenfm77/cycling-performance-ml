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
import requests
import re

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
    .alert-box { border-radius: 8px; padding: 14px 18px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

C = {
    "bg": "#0d1117", "panel": "#161b22", "grid": "#21262d",
    "text": "#c9d1d9", "muted": "#8b949e", "accent": "#58a6ff",
    "green": "#3fb950", "orange": "#f0883e", "red": "#f85149",
    "yellow": "#d29922", "purple": "#bc8cff",
}

STATE_COLORS = {
    "Undertrained": C["muted"], "Overreached": C["red"],
    "Deep Block": C["orange"], "Build Phase": C["yellow"],
    "Neutral": C["accent"], "Fresh": C["green"],
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
HOT_TEMP    = 28.0  # degrees C above which heat correction note appears


def _fetch_from_api() -> pd.DataFrame:
    """Fetch from Intervals.icu API — used on Streamlit Cloud when no local data."""
    try:
        athlete_id = st.secrets["INTERVALS_ATHLETE_ID"]
        api_key    = st.secrets["INTERVALS_API_KEY"].replace("API_KEY:", "").strip()
    except Exception:
        return pd.DataFrame()
    date_from = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%dT00:00:00")
    date_to   = datetime.now().strftime("%Y-%m-%dT23:59:59")
    try:
        r = requests.get(
            f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities",
            auth=("API_KEY", api_key),
            params={"oldest": date_from, "newest": date_to},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"API fetch failed: {e}")
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    df = pd.json_normalize(data)
    rename_map = {
        "start_date_local": "date", "moving_time": "duration_secs",
        "distance": "distance_m", "total_elevation_gain": "elevation",
        "average_watts": "power_avg", "weighted_average_watts": "power_np",
        "max_watts": "power_max", "average_heartrate": "hr_avg",
        "average_cadence": "cadence", "average_temp": "temp_avg",
        "icu_training_load": "tss", "icu_intensity": "if_score",
        "icu_eftp": "eftp", "icu_fitness": "ctl", "icu_fatigue": "atl",
        "icu_average_watts": "power_avg_icu", "icu_normalized_watts": "power_np_icu",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "duration_secs" in df.columns:
        df["duration_h"] = pd.to_numeric(df["duration_secs"], errors="coerce") / 3600
    if "power_avg" not in df.columns and "power_avg_icu" in df.columns:
        df["power_avg"] = df["power_avg_icu"]
    if "power_np" not in df.columns and "power_np_icu" in df.columns:
        df["power_np"] = df["power_np_icu"]
    if "power_np" not in df.columns:
        df["power_np"] = df.get("power_avg", pd.Series([np.nan]*len(df)))
    df["weight"] = 57.0

    # ── Derive training_type from the paired/planned workout label ────────────
    # Intervals.icu attaches the planned workout (the "Workout: FTP" tag you see
    # in the app). Its name is exposed on the activity record. The activity's own
    # "name" is the creative Strava title (e.g. "Arise and raze..."), so we must
    # look at the workout fields, not the activity name. We scan the most likely
    # fields for one of your known type labels and fall back to "—".
    KNOWN_TYPES = [
        "AEROBIC BASE", "VO2MAX", "VO2 MAX", "Q-I INTERVALS", "FATMAX",
        "PIRAMIDAL", "BILLAT", "TORQUE", "TEMPO", "END", "FTP", "SST",
    ]
    # Only trust real workout-label fields. Do NOT fall back to the activity
    # "name" — that is the creative Strava title and would false-match (e.g.
    # "around the bEND" -> END, "FRIENDLY tempo" -> TEMPO).
    candidate_cols = [
        "icu_workout_name", "workout_name", "workout_doc_name",
        "pairedWorkoutName", "paired_workout_name", "icu_workout", "workout",
    ]

    def _match_type(row) -> str:
        for col in candidate_cols:
            if col in row and isinstance(row[col], str) and row[col].strip():
                hay = " " + re.sub(r"[^A-Z0-9 ]", " ", row[col].upper())
                hay = re.sub(r"\s+", " ", hay) + " "
                for t in sorted(KNOWN_TYPES, key=len, reverse=True):
                    # normalize the label the same way as the haystack so
                    # hyphenated labels (Q-I INTERVALS) match correctly
                    tok = re.sub(r"[^A-Z0-9 ]", " ", t)
                    tok = re.sub(r"\s+", " ", tok).strip()
                    if f" {tok} " in hay:
                        return "VO2MAX" if t == "VO2 MAX" else t
        return "—"

    df["training_type"] = df.apply(_match_type, axis=1)
    return df


@st.cache_data(ttl=3600)
def load_data():
    # Column rename map shared by every Excel source
    excel_rename = {
        "Activity Date":          "date",
        "TRAINING_TYPE":          "training_type",
        "TSS":                    "tss",
        "IF":                     "if_score",
        "PowerAverage":           "power_avg",
        "Weighted Average Power": "power_np",
        "PowerMax":               "power_max",
        "TimeTotalInHours":       "duration_h",
        "HeartRateAverage":       "hr_avg",
        "WEIGHT_KG":              "weight",
        "Elevation Gain":         "elevation",
        "DistanceInMeters":       "distance_m",
        "Average Cadence":        "cadence",
        "Average Temperature":    "temp_avg",
        "Variability":            "variability",
    }

    df = None
    # ── Source 1: local files (your PC) ───────────────────────────────────────
    # intervals_api.py fetches from the API and writes combined_training_data.csv
    paths = ["data/combined_training_data.csv", "data/JOIN_STRAVA_TP.xlsx"]
    for p in paths:
        if Path(p).exists():
            if p.endswith(".csv"):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
                df = df.rename(columns=excel_rename)
            break

    # ── Source 2: live Intervals.icu API (Streamlit Cloud, no local CSV) ──────
    if df is None or len(df) == 0:
        df = _fetch_from_api()
    if df is None or len(df) == 0:
        st.error(
            "No data available. Add INTERVALS_ATHLETE_ID and INTERVALS_API_KEY "
            "to Streamlit Cloud secrets (app Settings → Secrets)."
        )
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Safe column creation — works whether data comes from Excel or API
    for col in ["tss", "power_avg", "hr_avg", "duration_h", "elevation",
                "if_score", "power_np", "power_max", "cadence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["tss"] = df["tss"].fillna(0)

    # Optional columns
    df["temp_avg"] = pd.to_numeric(df["temp_avg"], errors="coerce") if "temp_avg" in df.columns else np.nan
    df["weight"]   = pd.to_numeric(df["weight"],   errors="coerce").fillna(57.0) if "weight" in df.columns else pd.Series([57.0] * len(df))

    df["ctl"] = df["tss"].ewm(span=42, adjust=False).mean()
    df["atl"] = df["tss"].ewm(span=7,  adjust=False).mean()
    df["tsb"] = df["ctl"] - df["atl"]

    # Use normalized power (NP), not average power, for W/kg and efficiency.
    # Avg power is diluted by recovery valleys between intervals and is
    # misleading for interval-heavy rides. NP weights surges correctly.
    # For API-sourced rows power_np is often null (icu_normalized_watts not
    # always present); fall back to power_avg so the efficiency chart isn't
    # blank for recent sessions.
    df["w_per_kg"]    = df["power_np"] / df["weight"]
    _pwr_for_eff      = df["power_np"].fillna(df["power_avg"])
    df["efficiency"]  = np.where(df["hr_avg"] > 0, _pwr_for_eff / df["hr_avg"], np.nan)
    df["ftp_stimulus"]= (df["if_score"] ** 2) * df["duration_h"] * 100

    # ── Rolling efficiency metrics ────────────────────────────────────────────
    df["eff_28d"]    = df["efficiency"].rolling(28, min_periods=5).mean()
    df["eff_pct_vs_28d"] = (df["efficiency"] - df["eff_28d"]) / df["eff_28d"] * 100
    df["pwr_hr_28d"] = df["efficiency"].rolling(28, min_periods=5).mean()

    # ── Session quality score (0-100) ─────────────────────────────────────────
    # Combines: efficiency vs 28d avg + TSS + IF
    df["quality_score"] = np.nan
    valid = df["efficiency"].notna() & df["eff_28d"].notna()
    eff_std = max(float(df.loc[valid, "eff_28d"].std()), 0.01)
    tss_std = max(float(df.loc[valid, "tss"].std()), 0.01)
    if_std  = max(float(df.loc[valid, "if_score"].std()), 0.01)
    eff_z   = (df.loc[valid, "efficiency"] - df.loc[valid, "eff_28d"]) / eff_std
    tss_z   = (df.loc[valid, "tss"] - df.loc[valid, "tss"].mean()) / tss_std
    if_z    = (df.loc[valid, "if_score"] - df.loc[valid, "if_score"].mean()) / if_std
    raw_score = (eff_z * 0.5 + tss_z * 0.3 + if_z * 0.2)
    min_s, max_s = float(raw_score.min()), float(raw_score.max())
    if max_s > min_s:
        df.loc[valid, "quality_score"] = ((raw_score - min_s) / (max_s - min_s) * 100).clip(0, 100)

    # ── Norwegian Method compliance ───────────────────────────────────────────
    # Quality session = IF >= 0.85 (true threshold zone)
    df["is_quality"]      = df["if_score"] >= 0.85
    df["is_z3_drift"]     = (df["if_score"] >= 0.75) & (df["if_score"] < 0.85)
    df["is_true_z2"]      = df["if_score"] < 0.75
    df["is_hot_session"]  = df["temp_avg"].fillna(0) > HOT_TEMP

    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M")
    df["week"]  = df["date"].dt.to_period("W")

    conditions = [
        df["ctl"] < 30, df["tsb"] < -30, df["tsb"] < -10,
        df["tsb"] < 0,  df["tsb"] < 10,  df["tsb"] < 25,
    ]
    choices = ["Undertrained", "Overreached", "Deep Block",
               "Build Phase", "Neutral", "Fresh"]
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


df_all   = load_data()
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
    available_types = ["All"] + sorted(df_all["training_type"].dropna().unique().tolist())
    type_filter = st.multiselect("Training Types", available_types, default=["All"])
    st.markdown("---")
    st.markdown(f"**Last updated:** {datetime.now().strftime('%d %b %Y %H:%M')}")
    st.markdown(f"**Total sessions:** {len(df_all)}")
    st.markdown(f"**Date range:** {df_all['date'].min().date()} → {df_all['date'].max().date()}")

days_map = {
    "Last 30 days": 30, "Last 90 days": 90, "Last 6 months": 180,
    "Last 12 months": 365, "All time": 9999
}
cutoff      = datetime.now() - timedelta(days=days_map[date_range])
df          = df_all[df_all["date"] >= cutoff].copy()
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

now         = pd.Timestamp.now()
week_start  = now - timedelta(days=now.weekday())
last_week_s = week_start - timedelta(days=7)
this_week   = df_all[df_all["date"] >= week_start]
last_week   = df_all[(df_all["date"] >= last_week_s) & (df_all["date"] < week_start)]

def safe_sum(s):
    return pd.to_numeric(s, errors="coerce").sum()

def safe_mean(s):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return v.mean() if len(v) > 0 else 0

tw_tss  = safe_sum(this_week["tss"])
lw_tss  = safe_sum(last_week["tss"])
tw_elev = safe_sum(this_week["elevation"])
lw_elev = safe_sum(last_week["elevation"])
tw_if   = safe_mean(this_week["if_score"])
lw_if   = safe_mean(last_week["if_score"])
tw_pwr  = safe_mean(this_week["power_avg"])
lw_pwr  = safe_mean(last_week["power_avg"])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Weekly TSS", f"{tw_tss:.0f}", delta=f"{tw_tss - lw_tss:+.0f} vs last week")
with col2:
    st.metric("Sessions", f"{len(this_week)}", delta=f"{len(this_week)-len(last_week):+d} vs last week")
with col3:
    st.metric("Avg IF", f"{tw_if:.3f}" if tw_if > 0 else "—",
              delta=f"{tw_if-lw_if:+.3f}" if tw_if > 0 and lw_if > 0 else None)
with col4:
    st.metric("Avg Power", f"{tw_pwr:.0f}W" if tw_pwr > 0 else "—",
              delta=f"{tw_pwr-lw_pwr:+.0f}W" if tw_pwr > 0 and lw_pwr > 0 else None)
with col5:
    st.metric("Elevation", f"{tw_elev:.0f}m", delta=f"{tw_elev-lw_elev:+.0f}m vs last week")

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
            st.metric("eFTP", f"{ev:.0f}W", delta=f"{ev-ep:+.0f}W")
        else:
            st.metric("eFTP", "—")
    else:
        st.metric("eFTP", "—")
with col5:
    wkg_s   = df_all["w_per_kg"].dropna()
    wkg_val = wkg_s.tail(10).mean() if len(wkg_s) >= 10 else wkg_s.mean()
    st.metric("W/kg (NP, 10-session avg)", f"{wkg_val:.2f}")

st.markdown(
    f'<div class="state-box" style="background:{state_color}22; '
    f'border: 2px solid {state_color}; color:{state_color};">'
    f'Current State: {state}</div>',
    unsafe_allow_html=True
)
state_guide = {
    "Undertrained":      "⚠️ Build baseline volume — CTL too low",
    "Overreached":       "🚨 Rest immediately — TSB below -30",
    "Deep Block":        "💪 Hard training block — monitor recovery closely",
    "Build Phase":       "✅ Most productive training zone — keep pushing",
    "Neutral":           "⚖️ Balanced load — maintain or start taper",
    "Fresh":             "🟢 Race-ready window — quality sessions now",
    "Peak/Detrain Risk": "⚡ Peak form — race or start next block",
}
st.info(state_guide.get(state, ""))
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FATIGUE SIGNALS
# (replaces RPE — power/HR ratio is more reliable for this athlete)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔬 Objective Fatigue Signals")
st.caption(
    "Power/HR efficiency is more reliable than RPE for you — "
    "your perception is contaminated by heat, sleep and motivation. "
    "These metrics are not."
)

recent_eff = df_all[df_all["efficiency"].notna()].tail(60).copy()

col1, col2, col3 = st.columns(3)

# Current efficiency vs 28d average
curr_eff    = recent_eff["efficiency"].iloc[-1] if len(recent_eff) > 0 else np.nan
avg_eff_28d = recent_eff["eff_28d"].iloc[-1]   if len(recent_eff) > 0 else np.nan
eff_delta   = ((curr_eff - avg_eff_28d) / avg_eff_28d * 100) if avg_eff_28d else np.nan

with col1:
    color = C["green"] if eff_delta and eff_delta > 0 else C["red"] if eff_delta and eff_delta < -5 else C["yellow"]
    st.metric(
        "W/BPM vs 28d avg",
        f"{curr_eff:.2f}" if not np.isnan(curr_eff) else "—",
        delta=f"{eff_delta:+.1f}%" if eff_delta and not np.isnan(eff_delta) else None
    )

# Norwegian compliance this week
quality_this_week = this_week[this_week["if_score"] >= 0.85]
z3_drift_week     = this_week[(this_week["if_score"] >= 0.75) & (this_week["if_score"] < 0.85)]

with col2:
    norwegian_ok = len(quality_this_week)
    st.metric(
        "True Threshold Sessions",
        f"{norwegian_ok} / {len(this_week)}",
        delta="✅ Norwegian target" if norwegian_ok >= 2 else "⚠️ Need 2 quality sessions"
    )

with col3:
    z3_count = len(z3_drift_week)
    st.metric(
        "Z3 Drift Sessions (avoid)",
        f"{z3_count}",
        delta="✅ Clean" if z3_count == 0 else f"⚠️ {z3_count} sessions in grey zone"
    )

# Efficiency trend chart with heat annotation
fig_eff_trend = go.Figure()

# 28d rolling average band
fig_eff_trend.add_trace(go.Scatter(
    x=recent_eff["date"], y=recent_eff["eff_28d"],
    mode="lines", name="28d Rolling Avg",
    line=dict(color=C["accent"], width=2, dash="dash"),
    opacity=0.7
))

# Upper/lower bounds (±5%)
fig_eff_trend.add_trace(go.Scatter(
    x=recent_eff["date"],
    y=recent_eff["eff_28d"] * 1.05,
    mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip"
))
fig_eff_trend.add_trace(go.Scatter(
    x=recent_eff["date"],
    y=recent_eff["eff_28d"] * 0.95,
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
    name="±5% normal range", hoverinfo="skip"
))

# Session dots coloured by efficiency vs baseline
dot_colors = []
for _, row in recent_eff.iterrows():
    if pd.isna(row["efficiency"]) or pd.isna(row["eff_28d"]):
        dot_colors.append(C["muted"])
    elif row["efficiency"] >= row["eff_28d"] * 1.05:
        dot_colors.append(C["green"])
    elif row["efficiency"] <= row["eff_28d"] * 0.95:
        dot_colors.append(C["red"])
    else:
        dot_colors.append(C["yellow"])

fig_eff_trend.add_trace(go.Scatter(
    x=recent_eff["date"],
    y=recent_eff["efficiency"],
    mode="markers",
    name="Session W/BPM",
    marker=dict(color=dot_colors, size=8, opacity=0.85),
    text=[
        f"{row['date'].strftime('%d %b')}<br>"
        f"W/BPM: {row['efficiency']:.2f}<br>"
        f"Baseline: {row['eff_28d']:.2f}<br>"
        f"Temp: {row['temp_avg']:.0f}°C" if not pd.isna(row.get('temp_avg', np.nan)) else
        f"{row['date'].strftime('%d %b')}<br>W/BPM: {row['efficiency']:.2f}"
        for _, row in recent_eff.iterrows()
    ],
    hoverinfo="text"
))

# Mark hot sessions
hot = recent_eff[recent_eff["is_hot_session"] == True]
if len(hot) > 0:
    fig_eff_trend.add_trace(go.Scatter(
        x=hot["date"], y=hot["efficiency"],
        mode="markers", name=f"Hot session (>{HOT_TEMP}°C)",
        marker=dict(color=C["orange"], size=14, symbol="circle-open", line=dict(width=2)),
        hoverinfo="skip"
    ))

fig_eff_trend.update_layout(
    title="Cardiac Efficiency (W/BPM) — Objective Fatigue Tracker<br>"
          "<sup>🟢 Above baseline = adapting · 🔴 Below baseline = fatigued · 🟠 Circle = hot session</sup>",
    height=420, **PLOTLY_LAYOUT
)
st.plotly_chart(fig_eff_trend, use_container_width=True)

# Fatigue alert
if eff_delta and not np.isnan(eff_delta):
    if eff_delta < -5:
        st.markdown(
            f'<div class="alert-box" style="background:{C["red"]}22; '
            f'border:1px solid {C["red"]}; color:{C["text"]};">'
            f'🚨 <b>Fatigue Alert:</b> W/BPM is {abs(eff_delta):.1f}% below your 28-day baseline. '
            f'Reduce intensity this week regardless of how you feel subjectively.</div>',
            unsafe_allow_html=True
        )
    elif eff_delta > 5:
        st.markdown(
            f'<div class="alert-box" style="background:{C["green"]}22; '
            f'border:1px solid {C["green"]}; color:{C["text"]};">'
            f'✅ <b>Adaptation signal:</b> W/BPM is {eff_delta:.1f}% above your 28-day baseline. '
            f'Your body is responding well — this is a good week to push quality.</div>',
            unsafe_allow_html=True
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# NORWEGIAN METHOD COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h2><span style="display:inline-block;width:1.1em;height:0.8em;'
    'position:relative;vertical-align:-0.05em;margin-right:0.35em;'
    'background:#ba0c2f;border-radius:2px;overflow:hidden;">'
    '<span style="position:absolute;left:0.34em;top:0;width:0.18em;height:100%;'
    'background:#fff;"></span>'
    '<span style="position:absolute;left:0;top:0.31em;width:100%;height:0.18em;'
    'background:#fff;"></span>'
    '<span style="position:absolute;left:0.39em;top:0;width:0.08em;height:100%;'
    'background:#00205b;"></span>'
    '<span style="position:absolute;left:0;top:0.36em;width:100%;height:0.08em;'
    'background:#00205b;"></span>'
    '</span>Norwegian Method Compliance</h2>',
    unsafe_allow_html=True,
)
st.caption(
    "Double threshold: 2 sessions/week at true threshold (IF ≥ 0.85). "
    "Avoid Z3 drift (IF 0.75-0.85). Keep Z2 pure (IF < 0.75). No VO2max unless specifically planned."
)

# Weekly compliance over time
weekly_compliance = (
    df.groupby(pd.Grouper(key="date", freq="W"))
    .apply(lambda x: pd.Series({
        "quality_sessions": (x["if_score"] >= 0.85).sum(),
        "z3_drift":         ((x["if_score"] >= 0.75) & (x["if_score"] < 0.85)).sum(),
        "pure_z2":          (x["if_score"] < 0.75).sum(),
        "total":            len(x),
    }))
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig_norw = go.Figure()
    fig_norw.add_trace(go.Bar(
        x=weekly_compliance["date"], y=weekly_compliance["quality_sessions"],
        name="True Threshold (IF≥0.85)", marker_color=C["green"], opacity=0.85))
    fig_norw.add_trace(go.Bar(
        x=weekly_compliance["date"], y=weekly_compliance["z3_drift"],
        name="Z3 Drift — avoid (IF 0.75-0.85)", marker_color=C["orange"], opacity=0.85))
    fig_norw.add_hline(y=2, line_dash="dot", line_color=C["green"],
                        annotation_text="Norwegian target (2 sessions)")
    fig_norw.update_layout(
        barmode="stack",
        title="Weekly Session Quality Distribution",
        height=350, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_norw, use_container_width=True)

with col2:
    # IF distribution histogram
    if_data = df[df["if_score"].notna() & (df["if_score"] > 0.3)]
    fig_if_hist = go.Figure()
    fig_if_hist.add_trace(go.Histogram(
        x=if_data["if_score"], nbinsx=30,
        marker_color=C["accent"], opacity=0.75, name="Sessions"))
    fig_if_hist.add_vline(x=0.75, line_dash="dash", line_color=C["yellow"],
                           annotation_text="Z2/Z3 boundary (0.75)")
    fig_if_hist.add_vline(x=0.85, line_dash="dash", line_color=C["green"],
                           annotation_text="Threshold start (0.85)")
    fig_if_hist.add_vline(x=0.95, line_dash="dash", line_color=C["red"],
                           annotation_text="VO2max (0.95)")
    fig_if_hist.update_layout(
        title=f"IF Distribution — {date_range}<br>"
              f"<sup>Ideal: big spike at <0.75 (Z2) + smaller spike at 0.85-0.95 (threshold)</sup>",
        height=350, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_if_hist, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PMC CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📈 Performance Management Chart")

fig_pmc = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.4, 0.35, 0.25],
    subplot_titles=["CTL vs ATL (Fitness vs Fatigue)",
                    "TSB — Form / Freshness", "Weekly TSS"]
)
fig_pmc.add_trace(go.Scatter(x=df["date"], y=df["ctl"], name="CTL Fitness",
    line=dict(color=C["green"], width=2.5)), row=1, col=1)
fig_pmc.add_trace(go.Scatter(x=df["date"], y=df["atl"], name="ATL Fatigue",
    line=dict(color=C["orange"], width=2.5),
    fill="tonexty", fillcolor="rgba(240,136,62,0.1)"), row=1, col=1)
for sname, color in STATE_COLORS.items():
    mask = df["fatigue_state"] == sname
    if mask.sum() > 0:
        fig_pmc.add_trace(go.Scatter(
            x=df.loc[mask, "date"], y=df.loc[mask, "tsb"],
            mode="markers", name=sname,
            marker=dict(color=color, size=5, opacity=0.7)), row=2, col=1)
fig_pmc.add_hline(y=-30, line_dash="dash", line_color=C["red"],
                   annotation_text="-30 Overreach", row=2, col=1)
fig_pmc.add_hline(y=25, line_dash="dash", line_color=C["purple"],
                   annotation_text="+25 Peak", row=2, col=1)
fig_pmc.add_hline(y=0, line_dash="dot", line_color=C["muted"], row=2, col=1)
weekly_tss = df.resample("W", on="date")["tss"].sum().reset_index()
fig_pmc.add_trace(go.Bar(
    x=weekly_tss["date"], y=weekly_tss["tss"],
    marker_color=[C["red"] if t >= 700 else C["green"] if t >= 400 else C["yellow"]
                  for t in weekly_tss["tss"]],
    name="Weekly TSS", opacity=0.75), row=3, col=1)
fig_pmc.update_layout(height=750, showlegend=True,
    paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
    font=dict(color=C["text"]),
    legend=dict(bgcolor=C["panel"], bordercolor=C["grid"]))
for i in range(1, 4):
    fig_pmc.update_xaxes(gridcolor=C["grid"], row=i, col=1)
    fig_pmc.update_yaxes(gridcolor=C["grid"], row=i, col=1)
st.plotly_chart(fig_pmc, use_container_width=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FTP DEVELOPMENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 FTP Development Analysis")
st.caption("Is your training mix actually driving FTP up?")

col1, col2 = st.columns(2)
with col1:
    if len(df_main) > 0:
        stim = (df_main.groupby("training_type")["ftp_stimulus"]
                .mean().reset_index().sort_values("ftp_stimulus", ascending=True))
        stim["is_driver"] = stim["training_type"].isin(FTP_DRIVERS)
        fig_stim = go.Figure()
        fig_stim.add_trace(go.Bar(
            x=stim["ftp_stimulus"], y=stim["training_type"], orientation="h",
            marker_color=[C["green"] if d else C["muted"] for d in stim["is_driver"]],
            opacity=0.85, text=[f"{v:.1f}" for v in stim["ftp_stimulus"]],
            textposition="outside"))
        fig_stim.update_layout(
            title=f"FTP Stimulus Score — {date_range}",
            height=420, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_stim, use_container_width=True)
    else:
        st.info("No labelled training type data in this period.")

with col2:
    if len(df_main) > 0:
        driver_tss = df_main.loc[df_main["training_type"].isin(FTP_DRIVERS), "tss"].sum()
        base_tss   = df_main.loc[~df_main["training_type"].isin(FTP_DRIVERS), "tss"].sum()
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=["FTP Drivers", "Base Volume"],
            values=[driver_tss, base_tss],
            marker_colors=[C["green"], C["muted"]],
            hole=0.5, textinfo="label+percent"))
        fig_pie.update_layout(title="TSS Split — Quality vs Base Volume",
            height=420, paper_bgcolor=C["bg"],
            font=dict(color=C["text"]), legend=dict(bgcolor=C["panel"]))
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# POWER CURVE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ⚡ Power Curve & Best Efforts")

col1, col2 = st.columns(2)

# Real mean-maximal power curve, produced by intervals_api.py from the
# Intervals.icu power-curves (MMP) endpoint and saved to data/power_curve.csv.
# This replaces the old approximation that bucketed whole rides by their total
# duration (which produced a flat, meaningless curve).
pc_df = pd.DataFrame()
_pc_path = "data/power_curve.csv"
if Path(_pc_path).exists():
    try:
        pc_df = pd.read_csv(_pc_path)
    except Exception:
        pc_df = pd.DataFrame()

if pc_df.empty or "watts" not in pc_df.columns:
    with col1:
        st.info(
            "Power curve unavailable. Run `python src/intervals_api.py` to fetch "
            "your mean-maximal power from Intervals.icu, then redeploy."
        )
else:
    _WEIGHT = 56.8
    pc_df = pc_df.sort_values("secs").reset_index(drop=True)
    pc_df["wkg"] = pc_df["watts"] / _WEIGHT
    _tick_vals = pc_df["secs"].tolist()
    _tick_text = pc_df["duration"].tolist()
    with col1:
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Scatter(
            x=pc_df["secs"], y=pc_df["watts"],
            mode="lines+markers+text",
            line=dict(color=C["purple"], width=3),
            marker=dict(size=10, color=C["purple"]),
            text=[f"{w:.0f}W" for w in pc_df["watts"]],
            textposition="top center", name="Best Power (W)"))
        fig_pc.add_hline(y=240, line_dash="dot", line_color=C["yellow"],
                          annotation_text="Current FTP (240W)")
        fig_pc.add_hline(y=275, line_dash="dot", line_color=C["green"],
                          annotation_text="Target FTP (275W)", opacity=0.4)
        fig_pc.update_xaxes(tickvals=_tick_vals, ticktext=_tick_text)
        fig_pc.update_layout(title="Power Curve — Best Mean-Maximal Power",
            height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_pc, use_container_width=True)

    with col2:
        _ftp_wkg = round(240 / _WEIGHT, 2)
        _tgt_wkg = round(275 / _WEIGHT, 2)
        fig_wkg_curve = go.Figure()
        # Categorical x so all bars are evenly spaced regardless of duration
        fig_wkg_curve.add_trace(go.Bar(
            x=pc_df["duration"], y=pc_df["wkg"],
            marker_color=[C["purple"] if w >= 8.0 else C["red"] if w >= 5.0
                          else C["orange"] if w >= 4.0
                          else C["yellow"] if w >= 3.5 else C["accent"]
                          for w in pc_df["wkg"]],
            text=[f"{w:.2f}" for w in pc_df["wkg"]],
            textposition="outside", opacity=0.85))
        fig_wkg_curve.add_hline(y=_ftp_wkg, line_dash="dot", line_color=C["yellow"],
                                  annotation_text=f"Current FTP ({_ftp_wkg} W/kg)")
        fig_wkg_curve.add_hline(y=_tgt_wkg, line_dash="dot", line_color=C["green"],
                                  annotation_text=f"Target ({_tgt_wkg} W/kg)", opacity=0.4)
        _y_max = round(pc_df["wkg"].max() * 1.15, 1)
        fig_wkg_curve.update_yaxes(range=[0, _y_max])
        fig_wkg_curve.update_layout(title=f"W/kg at Each Duration — {_WEIGHT}kg Climber",
            height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_wkg_curve, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FTP PROGRESSION MONTH BY MONTH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📈 FTP Progression — Month by Month")

col1, col2 = st.columns(2)
with col1:
    monthly_ftp = (
        df_all[df_all["power_np"].fillna(0) > 0]
        .groupby(df_all["date"].dt.to_period("M"))
        .agg(best_np=("power_np", "max"), sessions=("tss", "count"))
        .reset_index()
    )
    monthly_ftp["month_dt"] = monthly_ftp["date"].apply(lambda p: p.start_time)
    monthly_ftp["ftp_est"]  = monthly_ftp["best_np"] * 0.95
    fig_ftp_prog = go.Figure()
    fig_ftp_prog.add_trace(go.Scatter(
        x=monthly_ftp["month_dt"], y=monthly_ftp["ftp_est"],
        mode="lines+markers", name="Est. FTP (best NP × 0.95)",
        line=dict(color=C["purple"], width=2.5), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(188,140,255,0.08)"))
    fig_ftp_prog.add_trace(go.Scatter(
        x=monthly_ftp["month_dt"],
        y=monthly_ftp["ftp_est"].rolling(3).mean(),
        mode="lines", name="3-month trend",
        line=dict(color=C["accent"], width=2, dash="dash")))
    fig_ftp_prog.add_vline(x=pd.Timestamp("2025-06-01"), line_dash="dash",
                            line_color=C["red"], opacity=0.7,
                            annotation_text="Surgery Jun 2025")
    fig_ftp_prog.add_hline(y=235, line_dash="dot", line_color=C["yellow"],
                            annotation_text="Current FTP (240W)")
    fig_ftp_prog.add_hline(y=275, line_dash="dot", line_color=C["green"],
                            annotation_text="Pre-accident FTP (275W)", opacity=0.4)
    fig_ftp_prog.update_layout(title="Monthly FTP Progression 2019–2026",
        height=400, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_ftp_prog, use_container_width=True)

with col2:
    annual_ftp = (
        df_all[df_all["power_np"].fillna(0) > 0]
        .groupby("year")
        .agg(peak_ftp=("power_np", lambda x: x.max() * 0.95),
             avg_wkg=("w_per_kg", "mean"))
        .reset_index()
    )
    fig_annual = go.Figure()
    fig_annual.add_trace(go.Bar(
        x=annual_ftp["year"].astype(str), y=annual_ftp["peak_ftp"],
        marker_color=[C["red"] if y == 2025 else C["purple"] for y in annual_ftp["year"]],
        opacity=0.85, text=[f"{v:.0f}W" for v in annual_ftp["peak_ftp"]],
        textposition="outside", name="Peak FTP estimate"))
    fig_annual.update_layout(
        title="Peak Estimated FTP by Year<br><sup>Red = surgery year</sup>",
        height=400, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_annual, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ELEVATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏔️ Elevation — Climber Stats")
st.caption("57kg climber — elevation per week is your key volume metric")

total_elev  = df_all["elevation"].sum()
avg_wk_elev = total_elev / max((df_all["date"].max()-df_all["date"].min()).days/7, 1)
best_elev   = df_all["elevation"].max()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Career Elevation", f"{total_elev:,.0f}m")
with col2:
    st.metric("Avg Elevation / Week", f"{avg_wk_elev:,.0f}m")
with col3:
    st.metric("Best Single Session", f"{best_elev:,.0f}m")

col1, col2 = st.columns(2)
with col1:
    weekly_elev = df.resample("W", on="date").agg(
        elevation=("elevation", "sum")).reset_index()
    weekly_elev["elevation"] = weekly_elev["elevation"].fillna(0)
    fig_elev = go.Figure()
    fig_elev.add_trace(go.Bar(
        x=weekly_elev["date"], y=weekly_elev["elevation"],
        marker_color=[C["green"] if e >= 3000 else C["yellow"] if e >= 1500 else C["muted"]
                      for e in weekly_elev["elevation"]],
        opacity=0.85, name="Weekly Elevation (m)"))
    fig_elev.add_hline(y=3000, line_dash="dot", line_color=C["green"],
                        annotation_text="3000m/week target")
    fig_elev.add_hline(y=1500, line_dash="dot", line_color=C["yellow"],
                        annotation_text="1500m minimum", opacity=0.5)
    fig_elev.update_layout(title=f"Weekly Elevation — {date_range}",
        height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_elev, use_container_width=True)

with col2:
    annual_elev = df_all.groupby("year").agg(
        total_elev=("elevation", "sum")).reset_index()
    fig_annual_elev = go.Figure()
    fig_annual_elev.add_trace(go.Bar(
        x=annual_elev["year"].astype(str), y=annual_elev["total_elev"],
        marker_color=[C["red"] if y == 2025 else C["accent"] for y in annual_elev["year"]],
        opacity=0.85,
        text=[f"{v/1000:.1f}K" for v in annual_elev["total_elev"]],
        textposition="outside"))
    fig_annual_elev.update_layout(
        title="Total Elevation per Year<br><sup>Red = surgery year</sup>",
        height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_annual_elev, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# THIS YEAR vs SAME PERIOD LAST YEAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 This Year vs Same Period Last Year")

current_year = datetime.now().year
today_md     = datetime.now().strftime("%m-%d")
this_yr      = df_all[df_all["year"] == current_year]
last_yr      = df_all[(df_all["year"] == current_year-1) &
                       (df_all["date"].dt.strftime("%m-%d") <= today_md)]

def yr_stats(d):
    return {
        "sessions":  len(d),
        "total_tss": safe_sum(d["tss"]),
        "avg_power": safe_mean(d["power_avg"]),
        "avg_wkg":   safe_mean(d["w_per_kg"]),
        "elevation": safe_sum(d["elevation"]),
        "hours":     safe_sum(d["duration_h"]),
    }

ty = yr_stats(this_yr)
ly = yr_stats(last_yr)

col1, col2, col3, col4, col5, col6 = st.columns(6)
for col, (label, tv, lv, unit, dec) in zip(
    [col1, col2, col3, col4, col5, col6],
    [("Sessions", ty["sessions"], ly["sessions"], "", 0),
     ("Total TSS", ty["total_tss"], ly["total_tss"], "", 0),
     ("Avg Power", ty["avg_power"], ly["avg_power"], "W", 1),
     ("Avg W/kg", ty["avg_wkg"], ly["avg_wkg"], "", 2),
     ("Elevation", ty["elevation"], ly["elevation"], "m", 0),
     ("Hours", ty["hours"], ly["hours"], "h", 1)]
):
    fmt = f"{{:.{dec}f}}{unit}"
    with col:
        st.metric(
            f"{label} {current_year}",
            fmt.format(tv) if tv and not np.isnan(float(tv)) else "—",
            delta=f"{tv-lv:+.{dec}f}{unit} vs {current_year-1}"
            if lv and not np.isnan(float(lv)) else None
        )

col1, col2 = st.columns(2)
with col1:
    mc = [{"month": pd.Timestamp(f"{current_year}-{m:02d}-01").strftime("%b"),
           "this_year": safe_sum(df_all[(df_all["year"]==current_year) &
                                        (df_all["date"].dt.month==m)]["tss"]),
           "last_year": safe_sum(df_all[(df_all["year"]==current_year-1) &
                                        (df_all["date"].dt.month==m)]["tss"])}
          for m in range(1, 13)]
    mc_df = pd.DataFrame(mc)
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(x=mc_df["month"], y=mc_df["last_year"],
        name=str(current_year-1), marker_color=C["muted"], opacity=0.6))
    fig_compare.add_trace(go.Bar(x=mc_df["month"], y=mc_df["this_year"],
        name=str(current_year), marker_color=C["accent"], opacity=0.85))
    fig_compare.update_layout(barmode="group",
        title=f"Monthly TSS — {current_year} vs {current_year-1}",
        height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_compare, use_container_width=True)

with col2:
    wc = [{"month": pd.Timestamp(f"{current_year}-{m:02d}-01").strftime("%b"),
           "this_year": safe_mean(df_all[(df_all["year"]==current_year) &
                                          (df_all["date"].dt.month==m)]["w_per_kg"]),
           "last_year": safe_mean(df_all[(df_all["year"]==current_year-1) &
                                          (df_all["date"].dt.month==m)]["w_per_kg"])}
          for m in range(1, 13)]
    wc_df = pd.DataFrame(wc)
    fig_wkg_c = go.Figure()
    fig_wkg_c.add_trace(go.Scatter(x=wc_df["month"], y=wc_df["last_year"],
        name=str(current_year-1),
        line=dict(color=C["muted"], width=2, dash="dash"), mode="lines+markers"))
    fig_wkg_c.add_trace(go.Scatter(x=wc_df["month"], y=wc_df["this_year"],
        name=str(current_year),
        line=dict(color=C["purple"], width=2.5), mode="lines+markers"))
    fig_wkg_c.update_layout(
        title=f"Monthly W/kg — {current_year} vs {current_year-1}",
        height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_wkg_c, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# POWER & EFFICIENCY TREND
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ⚡ Power & Efficiency Trend")

col1, col2 = st.columns(2)
with col1:
    monthly = df.resample("ME", on="date").agg(
        avg_wkg=("w_per_kg", "mean"), max_wkg=("w_per_kg", "max")).reset_index()
    fig_wkg = go.Figure()
    fig_wkg.add_trace(go.Scatter(x=monthly["date"], y=monthly["avg_wkg"],
        mode="lines+markers", name="Avg W/kg",
        line=dict(color=C["purple"], width=2.5), marker=dict(size=6)))
    fig_wkg.add_trace(go.Scatter(x=monthly["date"], y=monthly["max_wkg"],
        mode="lines", name="Max W/kg",
        line=dict(color=C["accent"], width=1.5, dash="dash"), opacity=0.6))
    fig_wkg.add_hline(y=4.21, line_dash="dot", line_color=C["yellow"],
                       annotation_text="Current FTP target (4.21 W/kg)")
    fig_wkg.add_hline(y=4.82, line_dash="dot", line_color=C["purple"],
                       annotation_text="Pre-accident target (4.82 W/kg)", opacity=0.4)
    fig_wkg.update_layout(title="Monthly W/kg Trend", height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_wkg, use_container_width=True)

with col2:
    monthly_eff = df.resample("ME", on="date").agg(
        avg_eff=("efficiency", "mean")).reset_index().dropna()
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(x=monthly_eff["date"], y=monthly_eff["avg_eff"],
        mode="lines+markers", name="W/BPM",
        line=dict(color=C["orange"], width=2.5), marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(240,136,62,0.1)"))
    fig_eff.update_layout(
        title="Cardiac Efficiency (W per BPM) — Higher = Better",
        height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_eff, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING TYPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏋️ Training Type Analysis")
st.caption("All-time data regardless of time filter")

col1, col2 = st.columns(2)
with col1:
    type_counts = df_main_all["training_type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]
    fig_types = px.bar(type_counts, x="count", y="type", orientation="h",
        color="type", title="Sessions per Training Type (All Time)",
        color_discrete_sequence=px.colors.qualitative.Set2)
    fig_types.update_layout(height=400, showlegend=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_types, use_container_width=True)

with col2:
    type_if = (df_main_all.groupby("training_type")["if_score"]
               .mean().reset_index().sort_values("if_score", ascending=True))
    fig_if = px.bar(type_if, x="if_score", y="training_type", orientation="h",
        color="if_score", title="Avg Intensity Factor by Type (All Time)",
        color_continuous_scale=[[0, C["accent"]], [0.5, C["yellow"]], [1, C["red"]]])
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

display_cols = [c for c in ["date", "training_type", "tss", "if_score", "power_avg",
    "duration_h", "hr_avg", "w_per_kg", "elevation", "temp_avg", "quality_score",
    "fatigue_state"] if c in df_all.columns]
recent = df_all[display_cols].tail(20).sort_values("date", ascending=False).copy()
recent["date"] = recent["date"].dt.strftime("%d %b %Y")
for col in ["tss", "if_score", "power_avg", "duration_h", "hr_avg",
            "w_per_kg", "elevation", "temp_avg", "quality_score"]:
    if col in recent.columns:
        recent[col] = pd.to_numeric(recent[col], errors="coerce").round(2)
if "training_type" in recent.columns:
    recent["training_type"] = recent["training_type"].fillna("—")
recent = recent.rename(columns={
    "date": "Date", "training_type": "Type", "tss": "TSS", "if_score": "IF",
    "power_avg": "Power (W)", "duration_h": "Hours", "hr_avg": "Avg HR",
    "w_per_kg": "W/kg", "elevation": "Elev (m)", "temp_avg": "Temp °C",
    "quality_score": "Quality", "fatigue_state": "State"})
st.dataframe(recent, use_container_width=True, height=450,
    column_config={
        "TSS": st.column_config.ProgressColumn("TSS", min_value=0, max_value=300, format="%d"),
        "IF": st.column_config.NumberColumn("IF", format="%.3f"),
        "W/kg": st.column_config.NumberColumn("W/kg", format="%.2f"),
        "Quality": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100, format="%.0f"),
    })
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# WELLNESS & HRV
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 💚 Wellness & HRV")
st.caption("HRV and resting HR are the earliest objective signs of accumulated fatigue — "
           "they respond before CTL/ATL do.")

has_hrv = (
    not wellness.empty
    and "hrv" in wellness.columns
    and wellness["hrv"].notna().sum() > 5
)
has_rhr = (
    not wellness.empty
    and "resting_hr" in wellness.columns
    and wellness["resting_hr"].notna().sum() > 5
    and float(wellness["resting_hr"].std()) > 0
)

if has_hrv or has_rhr:
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
                title="Heart Rate Variability (HRV) — Higher = More Recovered",
                height=300, **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig_hrv, use_container_width=True)
        else:
            st.info("No HRV data in wellness_data.csv yet. Log HRV daily in Intervals.icu.")
    with col2:
        if has_rhr:
            fig_rhr = go.Figure()
            fig_rhr.add_trace(go.Scatter(
                x=wellness_recent["date"],
                y=wellness_recent["resting_hr"],
                mode="lines+markers",
                line=dict(color=C["orange"], width=2),
                name="Resting HR",
                fill="tozeroy",
                fillcolor="rgba(240,136,62,0.08)"
            ))
            fig_rhr.update_layout(
                title="Resting Heart Rate — Lower = More Recovered",
                height=300, **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig_rhr, use_container_width=True)
        else:
            st.info("No resting HR data in wellness_data.csv yet. Log it daily in Intervals.icu.")
else:
    st.info(
        "No HRV or resting heart rate data logged yet. "
        "Enter these daily in Intervals.icu (Wellness section) to track recovery trends here."
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# NEXT WEEK RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🗓️ Next Week Recommendation")

tsb_now = float(df_all.iloc[-1]["tsb"])
ctl_now = float(df_all.iloc[-1]["ctl"])
atl_now = float(df_all.iloc[-1]["atl"])

# Override recommendation if efficiency alert is active
fatigue_override = eff_delta and not np.isnan(eff_delta) and eff_delta < -5

if fatigue_override:
    rc, rt, rx = C["red"], "🚨 Physiological Fatigue Detected", \
        f"W/BPM is {abs(eff_delta):.1f}% below baseline despite TSB = {tsb_now:+.1f}. " \
        "Your body is more fatigued than TSB suggests. Reduce load this week."
elif tsb_now < -30:
    rc, rt, rx = C["red"], "🚨 Rest or Very Easy Week", \
        f"TSB = {tsb_now:.1f} — Overreached. Max 3 easy Z2 sessions. No quality until TSB > -20."
elif tsb_now < -10:
    rc, rt, rx = C["orange"], "💪 Continue Hard Block", \
        f"TSB = {tsb_now:.1f} — Deep build. 2 quality sessions: Wednesday FTP + Saturday PIRAMIDAL."
elif tsb_now < 5:
    rc, rt, rx = C["yellow"], "✅ Standard Build Week", \
        f"TSB = {tsb_now:.1f} — Build phase. Wednesday FTP/SST + Saturday PIRAMIDAL with TEMPO blocks."
elif tsb_now < 20:
    rc, rt, rx = C["green"], "🟢 Push Hard This Week", \
        f"TSB = {tsb_now:.1f} — Fresh. 4x10min FTP Wednesday + long PIRAMIDAL Saturday."
else:
    rc, rt, rx = C["purple"], "⚡ Peak Form — Race or FTP Test", \
        f"TSB = {tsb_now:.1f} — Peak form. 20-min FTP test or hardest session of the block."

st.markdown(
    f'<div style="background:{rc}22; border: 2px solid {rc}; border-radius: 10px; '
    f'padding: 20px; margin: 10px 0;">'
    f'<h3 style="color:{rc}; margin:0 0 10px 0;">{rt}</h3>'
    f'<p style="color:#c9d1d9; margin:0; font-size:1.05rem;">{rx}</p>'
    f'<p style="color:#8b949e; margin:10px 0 0 0; font-size:0.85rem;">'
    f'CTL={ctl_now:.1f} · ATL={atl_now:.1f} · TSB={tsb_now:+.1f} · '
    f'W/BPM vs baseline: {eff_delta:+.1f}%' if eff_delta and not np.isnan(eff_delta)
    else f'CTL={ctl_now:.1f} · ATL={atl_now:.1f} · TSB={tsb_now:+.1f}'
    + f'</p></div>',
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown("*Run `python src/intervals_api.py` to fetch latest data · Then click Refresh Data*")
