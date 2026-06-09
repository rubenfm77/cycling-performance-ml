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
</style>
""", unsafe_allow_html=True)

C = {
    "bg": "#0d1117", "panel": "#161b22", "grid": "#21262d",
    "text": "#c9d1d9", "muted": "#8b949e", "accent": "#58a6ff",
    "green": "#3fb950", "orange": "#f0883e", "red": "#f85149",
    "yellow": "#d29922", "purple": "#bc8cff",
}

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
                })
            break

    if df is None:
        st.error("No data found. Run python src/intervals_api.py first.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # PMC using correct EWM
    df["tss"] = pd.to_numeric(df["tss"], errors="coerce").fillna(0)
    df["ctl"] = df["tss"].ewm(span=42, adjust=False).mean()
    df["atl"] = df["tss"].ewm(span=7, adjust=False).mean()
    df["tsb"] = df["ctl"] - df["atl"]

    # W/kg
    if "weight" in df.columns:
        weight = pd.to_numeric(df["weight"], errors="coerce").fillna(57.0)
    else:
        weight = pd.Series([57.0] * len(df))
    df["w_per_kg"] = pd.to_numeric(df["power_avg"], errors="coerce") / weight

    # Efficiency
    hr = pd.to_numeric(df["hr_avg"], errors="coerce")
    pwr = pd.to_numeric(df["power_avg"], errors="coerce")
    df["efficiency"] = np.where(hr > 0, pwr / hr, np.nan)

    # Fatigue state
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


# ── Load all data ─────────────────────────────────────────────────────────────
df_all = load_data()
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

# ── Apply time filter ─────────────────────────────────────────────────────────
days_map = {
    "Last 30 days": 30, "Last 90 days": 90,
    "Last 6 months": 180, "Last 12 months": 365, "All time": 9999
}
cutoff = datetime.now() - timedelta(days=days_map[date_range])
df = df_all[df_all["date"] >= cutoff].copy()

# Training type analysis always uses ALL TIME data
df_main_all = df_all[df_all["training_type"].isin(MAIN_TYPES)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🚴 Cycling Performance Dashboard")
st.markdown(f"*{date_range} · {len(df)} sessions · Catalonia*")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TODAY'S STATUS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Today's Status")

latest = df_all.iloc[-1]
prev = df_all.iloc[-8] if len(df_all) >= 8 else df_all.iloc[0]

ctl_now = latest["ctl"]
atl_now = latest["atl"]
tsb_now = latest["tsb"]
state = latest["fatigue_state"]
state_color = STATE_COLORS.get(state, C["accent"])

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "CTL (Fitness)", f"{ctl_now:.1f}",
        delta=f"{ctl_now - prev['ctl']:+.1f} vs 7d ago"
    )
with col2:
    st.metric(
        "ATL (Fatigue)", f"{atl_now:.1f}",
        delta=f"{atl_now - prev['atl']:+.1f} vs 7d ago"
    )
with col3:
    st.metric(
        "TSB (Form)", f"{tsb_now:+.1f}",
        delta=f"{tsb_now - prev['tsb']:+.1f} vs 7d ago"
    )
with col4:
    if "eftp" in df_all.columns:
        eftp_series = pd.to_numeric(df_all["eftp"], errors="coerce").dropna()
        if len(eftp_series) > 0:
            eftp_val = eftp_series.iloc[-1]
            eftp_prev = eftp_series.iloc[-2] if len(eftp_series) > 1 else eftp_val
            st.metric("eFTP", f"{eftp_val:.0f}W",
                      delta=f"{eftp_val - eftp_prev:+.0f}W")
        else:
            st.metric("eFTP", "—")
    else:
        st.metric("eFTP", "—")
with col5:
    wkg_series = pd.to_numeric(df_all["w_per_kg"], errors="coerce").dropna()
    wkg_val = wkg_series.tail(10).mean() if len(wkg_series) >= 10 else wkg_series.mean()
    st.metric("W/kg (10-session avg)", f"{wkg_val:.2f}")

st.markdown(
    f'<div class="state-box" style="background:{state_color}22; '
    f'border: 2px solid {state_color}; color:{state_color};">'
    f'Current State: {state}</div>',
    unsafe_allow_html=True
)

state_guide = {
    "Undertrained": "⚠️ Build baseline volume — CTL too low",
    "Overreached": "🚨 Rest immediately — TSB below -30",
    "Deep Block": "💪 Hard training block — monitor recovery closely",
    "Build Phase": "✅ Most productive training zone — keep pushing",
    "Neutral": "⚖️ Balanced load — maintain or start taper",
    "Fresh": "🟢 Race-ready window — quality sessions now",
    "Peak/Detrain Risk": "⚡ Peak form — race or start next block",
}
st.info(state_guide.get(state, ""))
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
bar_colors = [
    C["red"] if t >= 700 else C["green"] if t >= 400 else C["yellow"]
    for t in weekly["tss"]
]
fig_pmc.add_trace(go.Bar(
    x=weekly["date"], y=weekly["tss"],
    marker_color=bar_colors, name="Weekly TSS", opacity=0.75
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
        y=4.12, line_dash="dot", line_color=C["yellow"],
        annotation_text="Current FTP target (4.12 W/kg = 235W)"
    )
    fig_wkg.add_hline(
        y=4.82, line_dash="dot", line_color=C["purple"],
        annotation_text="Pre-accident target (4.82 W/kg = 275W)",
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
# TRAINING TYPE ANALYSIS — always uses ALL TIME data
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Training Type Analysis")
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
# RECENT SESSIONS TABLE
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

# Fill missing training type from API data
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
# WELLNESS & HRV — only show if real HRV data exists
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
    rec_color = C["red"]
    rec_title = "🚨 Rest or Very Easy Week"
    rec_text = (
        f"TSB = {tsb_now:.1f} — You are overreached. "
        "Maximum 3 easy Z2 sessions, no quality work until TSB > -20."
    )
elif tsb_now < -10:
    rec_color = C["orange"]
    rec_title = "💪 Continue Hard Block"
    rec_text = (
        f"TSB = {tsb_now:.1f} — Deep build phase. "
        "Maintain 2 quality sessions. Wednesday FTP intervals + Saturday PIRAMIDAL."
    )
elif tsb_now < 5:
    rec_color = C["yellow"]
    rec_title = "✅ Standard Build Week"
    rec_text = (
        f"TSB = {tsb_now:.1f} — Build phase. "
        "Two quality sessions: Wednesday FTP/SST + Saturday PIRAMIDAL with TEMPO blocks."
    )
elif tsb_now < 20:
    rec_color = C["green"]
    rec_title = "🟢 Push Hard This Week"
    rec_text = (
        f"TSB = {tsb_now:.1f} — Fresh and ready. "
        "Increase intensity. Consider 4x10min FTP Wednesday + long PIRAMIDAL Saturday."
    )
else:
    rec_color = C["purple"]
    rec_title = "⚡ Peak Form — Race or Test"
    rec_text = (
        f"TSB = {tsb_now:.1f} — You are at peak form. "
        "Do a 20-min FTP test or target your hardest session of the block."
    )

st.markdown(
    f'<div style="background:{rec_color}22; border: 2px solid {rec_color}; '
    f'border-radius: 10px; padding: 20px; margin: 10px 0;">'
    f'<h3 style="color:{rec_color}; margin:0 0 10px 0;">{rec_title}</h3>'
    f'<p style="color:#c9d1d9; margin:0; font-size:1.05rem;">{rec_text}</p>'
    f'<p style="color:#8b949e; margin:10px 0 0 0; font-size:0.85rem;">'
    f'CTL={ctl_now:.1f} · ATL={atl_now:.1f} · TSB={tsb_now:+.1f}</p>'
    f'</div>',
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    "*Dashboard auto-refreshes every hour · "
    "Run `python src/intervals_api.py` to fetch latest data*"
)
