# src/pmc.py
"""
Performance Management Chart (PMC).
CTL · ATL · TSB · Fatigue state · Ramp rate analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from src.config import PALETTE, PLOT_STYLE, TSB_ZONES


def plot_pmc(df: pd.DataFrame, save_dir: str = "outputs") -> None:
    """Full 3-panel PMC: CTL/ATL · TSB · Weekly TSS."""

    plt.rcParams.update(PLOT_STYLE)
    P = PALETTE

    # Weekly aggregation for cleaner signal
    weekly = (
        df.groupby(df["date"].dt.to_period("W"))
        .agg(tss_sum=("tss", "sum"), ctl=("ctl", "last"),
             atl=("atl", "last"), tsb=("tsb", "last"),
             state=("fatigue_state", "last"))
        .reset_index()
    )
    weekly["week_dt"] = weekly["date"].apply(lambda p: p.start_time)

    state_colors = {
        "Undertrained":    P["muted"],
        "Overreached":     P["red"],
        "Deep Block":      P["orange"],
        "Build Phase":     P["yellow"],
        "Neutral":         P["accent"],
        "Fresh":           P["green"],
        "Peak/Detrain Risk": P["purple"],
    }

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=False)
    fig.suptitle("Performance Management Chart  |  2019–2026",
                 fontsize=15, fontweight="bold", color=P["accent"])

    # ── Panel 1: CTL / ATL ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(weekly["week_dt"], weekly["ctl"],
            color=P["green"], lw=2.5, label="CTL – Fitness (42d EWM)")
    ax.plot(weekly["week_dt"], weekly["atl"],
            color=P["orange"], lw=2.5, label="ATL – Fatigue (7d EWM)")
    ax.fill_between(weekly["week_dt"], weekly["ctl"], weekly["atl"],
                    where=weekly["atl"] > weekly["ctl"],
                    color=P["orange"], alpha=0.15, label="Fatigue > Fitness")
    ax.set_ylabel("Load (TSS units)", color=P["text"])
    ax.legend(fontsize=9)
    ax.set_title("Fitness vs Fatigue (CTL / ATL)", fontweight="bold")
    ax.grid(alpha=0.25)

    # ── Panel 2: TSB coloured by state ────────────────────────────────────────
    ax = axes[1]
    for state, sc in state_colors.items():
        mask = weekly["state"] == state
        ax.scatter(weekly.loc[mask, "week_dt"], weekly.loc[mask, "tsb"],
                   color=sc, s=12, alpha=0.8, label=state, zorder=3)
    ax.plot(weekly["week_dt"], weekly["tsb"],
            color=P["accent"], lw=1, alpha=0.3, zorder=1)
    ax.axhline(0,   color=P["muted"], lw=0.8, ls="--")
    ax.axhline(-30, color=P["red"],   lw=1.2, ls="--", label="_Overreach (-30)")
    ax.axhline(25,  color=P["purple"],lw=1.2, ls="--", label="_Peak (+25)")
    ax.fill_between(weekly["week_dt"], -30, -80,
                    alpha=0.06, color=P["red"])
    ax.fill_between(weekly["week_dt"], 25, 60,
                    alpha=0.06, color=P["purple"])
    ax.set_ylim(-65, 55)
    ax.set_ylabel("Form (TSB)", color=P["text"])
    ax.legend(fontsize=7.5, ncol=4, loc="lower right")
    ax.set_title("Training Stress Balance (Form)", fontweight="bold")
    ax.grid(alpha=0.25)

    # ── Panel 3: Weekly TSS bars ──────────────────────────────────────────────
    ax = axes[2]
    bar_colors = [
        P["red"] if t >= 700 else P["green"] if t >= 400 else P["yellow"]
        for t in weekly["tss_sum"]
    ]
    ax.bar(weekly["week_dt"], weekly["tss_sum"],
           color=bar_colors, width=5, alpha=0.75)
    ax.axhline(700, color=P["red"],    lw=1, ls="--", label="Overreach risk (700)")
    ax.axhline(400, color=P["orange"], lw=1, ls="--", label="High week (400)")
    ax.set_ylabel("Weekly TSS", color=P["text"])
    ax.set_xlabel("Date", color=P["text"])
    ax.legend(fontsize=8)
    ax.set_title("Weekly TSS Distribution", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    Path(save_dir).mkdir(exist_ok=True)
    out = Path(save_dir) / "pmc_full.png"
    plt.savefig(out, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  ✅ PMC saved → {out}")


def fatigue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return state distribution across all sessions."""
    dist = (df["fatigue_state"]
            .value_counts()
            .reset_index())
    dist.columns = ["state", "sessions"]
    dist["pct"] = (dist["sessions"] / dist["sessions"].sum() * 100).round(1)
    return dist


def ramp_rate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Flag weeks where ramp rate > 1.5 (injury risk threshold)."""
    weekly = (
        df.groupby(df["date"].dt.to_period("W"))
        .agg(ctl=("ctl", "last"), atl=("atl", "last"),
             tsb=("tsb", "last"), tss=("tss", "sum"))
        .reset_index()
    )
    weekly["ramp_rate"] = np.where(
        weekly["ctl"] > 0, weekly["atl"] / weekly["ctl"], np.nan
    )
    weekly["risk"] = weekly["ramp_rate"].apply(
        lambda r: "⚠️ Too Fast" if r > 1.5
        else "📈 Building" if r > 1.2
        else "✅ Balanced" if r > 0.8
        else "📉 Reducing"
    )
    return weekly


if __name__ == "__main__":
    from src.data_loader import load
    df = load()
    plot_pmc(df)
    print("\nFatigue state distribution:")
    print(fatigue_summary(df).to_string(index=False))
    print("\nRamp rate (last 10 weeks):")
    print(ramp_rate_analysis(df).tail(10)[
        ["date", "ctl", "atl", "tsb", "ramp_rate", "risk"]
    ].to_string(index=False))
