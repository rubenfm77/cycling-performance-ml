# src/monthly_composition_analysis.py
"""
Training composition analysis: does the MIX of training types over a 4-week
window predict better FTP outcomes than any single dominant type?

Two window strategies:
  - Calendar month  (non-overlapping, ~72 windows over 6 years — use for stats)
  - Rolling 28-day stepped by 7 days  (highly correlated; use for visual trends)

Outcome metric: next-window FTP proxy  =  max(NP or avg_power) × 0.95 in the
following period, minus this period's proxy.  Because we have no structured
FTP test log, this is an approximation — treat it as a signal, not ground truth.

⚠  Small-sample caveat: ~72 independent calendar-month windows over 6 years
gives roughly 60 data points once you remove the last month (no next-window).
Correlations and the RF are exploratory — they generate hypotheses rather than
confirm them.  Interpret accordingly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from scipy.stats import pearsonr, spearmanr

from src.config import MAIN_TYPES, PALETTE, PLOT_STYLE, TYPE_COLOURS

FTP_DRIVERS = ["FTP", "SST", "TEMPO", "PIRAMIDAL", "VO2MAX", "BILLAT", "Q-I INTERVALS"]

# Stable column name for each training type (replaces spaces/hyphens with _)
def _pct_col(t: str) -> str:
    return "pct_" + t.replace(" ", "_").replace("-", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Window-level feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _window_features(main_grp: pd.DataFrame, period_label: str) -> dict:
    """
    Compute composition features from a group of MAIN_TYPE sessions.
    Returns a dict ready to become a DataFrame row.
    """
    total_tss = float(main_grp["tss"].sum())
    total_h   = float(main_grp["duration_h"].sum()) if "duration_h" in main_grp.columns else np.nan
    n_sessions = int(len(main_grp))

    tss_by_type = main_grp.groupby("training_type")["tss"].sum()

    row: dict = {
        "period":      period_label,
        "total_tss":   total_tss,
        "total_hours": total_h,
        "n_sessions":  n_sessions,
        "n_types":     int(main_grp["training_type"].nunique()),
    }

    # % TSS by each training type
    for t in MAIN_TYPES:
        row[_pct_col(t)] = (
            float(tss_by_type.get(t, 0) / total_tss * 100) if total_tss > 0 else 0.0
        )

    # Quality TSS (FTP-driver types)
    q_tss = float(main_grp.loc[main_grp["training_type"].isin(FTP_DRIVERS), "tss"].sum())
    row["quality_tss"]        = q_tss
    row["quality_tss_pct"]    = q_tss / total_tss * 100 if total_tss > 0 else 0.0
    row["avg_ftp_stimulus"]   = float(main_grp["ftp_stimulus"].mean()) if "ftp_stimulus" in main_grp.columns else np.nan
    row["total_ftp_stimulus"] = float(main_grp["ftp_stimulus"].sum())  if "ftp_stimulus" in main_grp.columns else np.nan

    # Dominant type across all TSS
    dom_type = tss_by_type.idxmax() if total_tss > 0 and len(tss_by_type) > 0 else "—"
    dom_pct  = float(tss_by_type.max() / total_tss * 100) if total_tss > 0 else 0.0
    row["dominant_type"]     = dom_type
    row["dominant_type_pct"] = dom_pct

    # Dominant type among quality (FTP-driver) TSS only
    q_tss_by_type = (
        main_grp[main_grp["training_type"].isin(FTP_DRIVERS)]
        .groupby("training_type")["tss"].sum()
    )
    if len(q_tss_by_type) > 0 and q_tss > 0:
        qdom_type = q_tss_by_type.idxmax()
        qdom_pct  = float(q_tss_by_type.max() / q_tss * 100)
    else:
        qdom_type, qdom_pct = "—", 0.0
    row["quality_dominant_type"]     = qdom_type
    row["quality_dominant_type_pct"] = qdom_pct

    # Pattern label (based on quality TSS dominance)
    if qdom_pct > 50 and qdom_type != "—":
        row["pattern"] = f"Single: {qdom_type}"
    elif len(q_tss_by_type) >= 2:
        top2 = q_tss_by_type.nlargest(2).index.tolist()
        row["pattern"] = "Mixed: " + "+".join(sorted(top2))
    elif len(q_tss_by_type) == 1:
        row["pattern"] = f"Single: {q_tss_by_type.index[0]}"
    else:
        row["pattern"] = "No quality sessions"

    # Combination signature: top-3 types by TSS share (all types, not just drivers)
    top3 = tss_by_type.nlargest(3).index.tolist()
    row["combo_signature"] = "+".join(sorted(top3))

    return row


def _ftp_proxy_series(df: pd.DataFrame, period_col: str = "_month") -> pd.Series:
    """
    Best (NP or avg_power) × 0.95 per period, computed from ALL sessions.
    Returns a Series indexed by period.
    """
    pwr = df["power_np"].fillna(df["power_avg"]) if "power_np" in df.columns else df["power_avg"]
    df2 = df.copy()
    df2["_pwr"] = pd.to_numeric(pwr, errors="coerce")
    valid = df2[df2["_pwr"].fillna(0) > 50]
    return (valid.groupby(period_col)["_pwr"].max() * 0.95).rename("ftp_proxy")


# ─────────────────────────────────────────────────────────────────────────────
# Window builders
# ─────────────────────────────────────────────────────────────────────────────

def build_calendar_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar month.  Composition from MAIN_TYPE sessions;
    FTP proxy from ALL sessions; CTL/TSB context from last session of month.
    """
    df = df.copy()
    df["_month"] = df["date"].dt.to_period("M")
    df_main = df[df["training_type"].isin(MAIN_TYPES)].copy()

    ftp_series = _ftp_proxy_series(df, "_month")

    rows = []
    for period, grp in df_main.groupby("_month"):
        row = _window_features(grp, str(period))
        row["ftp_proxy"] = float(ftp_series.get(period, np.nan))
        month_all = df[df["_month"] == period]
        row["ctl_end"] = float(month_all["ctl"].iloc[-1]) if len(month_all) > 0 and "ctl" in month_all.columns else np.nan
        row["tsb_end"] = float(month_all["tsb"].iloc[-1]) if len(month_all) > 0 and "tsb" in month_all.columns else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    windows = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    windows["ftp_proxy_next"] = windows["ftp_proxy"].shift(-1)
    windows["ftp_gain"]       = windows["ftp_proxy_next"] - windows["ftp_proxy"]
    return windows


def build_rolling_windows(df: pd.DataFrame,
                           window_days: int = 28,
                           step_days: int = 7) -> pd.DataFrame:
    """
    Rolling 4-week windows stepped by step_days.
    Consecutive windows overlap heavily — do NOT use for significance tests.
    Useful for visual trends and seeing how composition shifted over time.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df_main = df[df["training_type"].isin(MAIN_TYPES)].copy()

    start_date = df_main["date"].min()
    end_date   = df_main["date"].max()

    rows = []
    current = start_date
    while current + pd.Timedelta(days=window_days // 2) <= end_date:
        wend = current + pd.Timedelta(days=window_days - 1)
        grp  = df_main[(df_main["date"] >= current) & (df_main["date"] <= wend)]
        if len(grp) >= 3:
            row = _window_features(grp, current.strftime("%Y-%m-%d"))
            row["window_start"] = current
            row["window_end"]   = wend
            # FTP proxy for this window
            all_in_window = df[(df["date"] >= current) & (df["date"] <= wend)]
            pwr_col = "power_np" if "power_np" in all_in_window.columns else "power_avg"
            pwr = all_in_window["power_np"].fillna(all_in_window["power_avg"]) if "power_np" in all_in_window.columns else all_in_window["power_avg"]
            max_pwr = pd.to_numeric(pwr, errors="coerce").max()
            row["ftp_proxy"] = float(max_pwr * 0.95) if pd.notna(max_pwr) and max_pwr > 50 else np.nan
            row["ctl_end"]   = float(all_in_window["ctl"].iloc[-1]) if len(all_in_window) > 0 and "ctl" in all_in_window.columns else np.nan
            row["tsb_end"]   = float(all_in_window["tsb"].iloc[-1]) if len(all_in_window) > 0 and "tsb" in all_in_window.columns else np.nan
            rows.append(row)
        current += pd.Timedelta(days=step_days)

    if not rows:
        return pd.DataFrame()

    rolling = pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)
    rolling["ftp_proxy_next"] = rolling["ftp_proxy"].shift(-1)
    rolling["ftp_gain"]       = rolling["ftp_proxy_next"] - rolling["ftp_proxy"]
    return rolling


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def classify_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Add is_single_dominant and is_mixed boolean columns."""
    out = windows.copy()
    out["is_single_dominant"] = out["quality_dominant_type_pct"] > 50
    out["is_mixed"] = (out["quality_dominant_type_pct"] <= 50) & (out["quality_tss_pct"] > 10)
    return out


def composition_correlations(windows: pd.DataFrame) -> pd.DataFrame:
    """Pearson + Spearman correlations between % TSS features and next-month ftp_gain."""
    valid = windows[windows["ftp_gain"].notna() & (windows["total_tss"] > 50)].copy()
    if len(valid) < 10:
        return pd.DataFrame()

    pct_cols  = [c for c in valid.columns if c.startswith("pct_")]
    extra     = ["total_tss", "quality_tss_pct", "n_sessions", "n_types",
                 "total_ftp_stimulus", "ctl_end", "tsb_end"]
    feat_cols = pct_cols + [c for c in extra if c in valid.columns]

    rows = []
    for col in feat_cols:
        sub = valid[[col, "ftp_gain"]].dropna()
        if len(sub) < 8:
            continue
        try:
            r_p, p_p = pearsonr(sub[col], sub["ftp_gain"])
            r_s, p_s = spearmanr(sub[col], sub["ftp_gain"])
        except Exception:
            continue
        rows.append({
            "feature":    col,
            "pearson_r":  round(r_p, 3),
            "pearson_p":  round(p_p, 4),
            "spearman_r": round(r_s, 3),
            "spearman_p": round(p_s, 4),
            "n":          len(sub),
            "sig": "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns",
        })

    return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)


def run_composition_rf(windows: pd.DataFrame) -> dict:
    """
    Random Forest regressor: composition % features → next-month ftp_gain.
    Returns None if sklearn unavailable or too little data.
    """
    if not HAS_SKLEARN:
        return {}

    valid = windows[windows["ftp_gain"].notna() & (windows["total_tss"] > 50)].copy()
    pct_cols  = [c for c in valid.columns if c.startswith("pct_")]
    extra     = ["total_tss", "quality_tss_pct", "n_sessions", "total_ftp_stimulus",
                 "ctl_end", "tsb_end"]
    feat_cols = pct_cols + [c for c in extra if c in valid.columns]

    sub = valid[feat_cols + ["ftp_gain"]].dropna()
    if len(sub) < 20:
        print(f"  ⚠️  Only {len(sub)} windows with valid outcome — RF skipped (need ≥20)")
        return {}

    X = sub[feat_cols]
    y = sub["ftp_gain"]

    reg = RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=max(3, len(sub)//10),
        random_state=42, n_jobs=-1
    )
    n_folds  = min(5, len(sub) // 5)
    cv_scores = cross_val_score(reg, X, y, cv=n_folds, scoring="r2")
    reg.fit(X, y)

    imp = pd.Series(reg.feature_importances_, index=feat_cols).sort_values(ascending=False)

    print(f"\n  Composition RF  R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
          f"(n={len(sub)} windows, {n_folds}-fold — exploratory)")
    print("  Top composition features:")
    for feat, v in imp.head(6).items():
        print(f"    {feat:<40} {v:.3f}")

    return {"model": reg, "importance": imp, "cv_r2": cv_scores, "n": len(sub)}


def top_combinations(windows: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Rank combo_signatures by average next-month ftp_gain."""
    valid = windows[windows["ftp_gain"].notna()].copy()
    ranked = (
        valid.groupby("combo_signature")
        .agg(
            n_months     = ("ftp_gain", "count"),
            avg_ftp_gain = ("ftp_gain", "mean"),
            median_gain  = ("ftp_gain", "median"),
            avg_tss      = ("total_tss", "mean"),
            avg_ctl      = ("ctl_end", "mean"),
            avg_tsb      = ("tsb_end", "mean"),
        )
        .reset_index()
        .sort_values("avg_ftp_gain", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"]     = range(1, len(ranked) + 1)
    ranked["reliable"] = ranked["n_months"] >= 3
    return ranked.head(n)


def mixed_vs_single_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """Average outcome by pattern (Single: X vs Mixed: X+Y vs No quality)."""
    valid = windows[windows["ftp_gain"].notna()].copy()
    return (
        valid.groupby("pattern")
        .agg(
            n_months     = ("ftp_gain", "count"),
            avg_ftp_gain = ("ftp_gain", "mean"),
            median_gain  = ("ftp_gain", "median"),
            avg_tss      = ("total_tss", "mean"),
            avg_tsb      = ("tsb_end", "mean"),
        )
        .reset_index()
        .sort_values("avg_ftp_gain", ascending=False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(windows: pd.DataFrame, corr: pd.DataFrame,
                  rf: dict, combos: pd.DataFrame,
                  mixed_summary: pd.DataFrame) -> None:

    n_windows = len(windows)
    n_valid   = int(windows["ftp_gain"].notna().sum())

    print(f"\n{'='*65}")
    print("  TRAINING COMPOSITION ANALYSIS — CALENDAR-MONTH WINDOWS")
    print(f"{'='*65}")
    print(f"  Total monthly windows : {n_windows}")
    print(f"  Windows with outcome  : {n_valid}  (last month has no next-month yet)")
    print(f"\n  ⚠️  Small-sample caveat")
    print(f"      {n_valid} independent observations.  Any correlation with |r| < 0.3")
    print(f"      or p > 0.10 should be treated as noise at this sample size.")
    print(f"      The RF is exploratory — it identifies which composition features")
    print(f"      *might* matter, not which ones *do* matter.")

    if len(corr) > 0:
        print(f"\n  Top correlations (% TSS by type → next-month FTP gain):")
        print(f"  {'Feature':<42} {'Pearson':>8} {'Spearman':>9} {'Sig':>5} {'n':>5}")
        print(f"  {'-'*67}")
        for _, row in corr.head(10).iterrows():
            print(f"  {row['feature']:<42} {row['pearson_r']:>8.3f} "
                  f"{row['spearman_r']:>9.3f} {row['sig']:>5} {row['n']:>5}")

    print(f"\n  Mixed vs single-dominant windows:")
    print(f"  {'Pattern':<35} {'n':>4} {'Avg gain':>9} {'Median':>8} {'Avg TSS':>9}")
    print(f"  {'-'*65}")
    for _, r in mixed_summary.iterrows():
        print(f"  {r['pattern']:<35} {r['n_months']:>4} "
              f"{r['avg_ftp_gain']:>9.1f} {r['median_gain']:>8.1f} {r['avg_tss']:>9.0f}")

    print(f"\n  Top type combinations (by avg next-month FTP gain):")
    print(f"  {'Rank':>4} {'Combination':<40} {'n':>4} {'Avg gain':>9} {'Reliable':>9}")
    print(f"  {'-'*70}")
    for _, r in combos.head(12).iterrows():
        flag = "✓" if r["reliable"] else "⚠ n<3"
        print(f"  {r['rank']:>4} {r['combo_signature']:<40} {r['n_months']:>4} "
              f"{r['avg_ftp_gain']:>9.1f} {flag:>9}")


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_composition(windows: pd.DataFrame, df_all: pd.DataFrame,
                     save_dir: str = "outputs") -> None:
    """
    2-panel figure:
    Top   — stacked bar: % TSS by training type per calendar month
    Bottom — FTP proxy trend so peaks can be correlated visually with composition above
    """
    if not HAS_MPL:
        print("  matplotlib unavailable — skipping composition chart")
        return

    plt.rcParams.update(PLOT_STYLE)
    P = PALETTE

    # Monthly FTP proxy from all sessions
    _pwr = (df_all["power_np"].fillna(df_all["power_avg"])
            if "power_np" in df_all.columns else df_all["power_avg"])
    df_all = df_all.copy()
    df_all["_pwr"]   = pd.to_numeric(_pwr, errors="coerce")
    df_all["_month"] = df_all["date"].dt.to_period("M")
    monthly_ftp = (
        df_all[df_all["_pwr"].fillna(0) > 50]
        .groupby("_month")["_pwr"]
        .max()
        .reset_index()
    )
    monthly_ftp.columns = ["_month", "best_pwr"]
    monthly_ftp["ftp_est"]   = monthly_ftp["best_pwr"] * 0.95
    monthly_ftp["month_dt"]  = monthly_ftp["_month"].apply(lambda p: p.start_time)
    monthly_ftp["ftp_trend"] = monthly_ftp["ftp_est"].rolling(3).mean()

    # Filter windows to months with enough TSS
    wp = windows[windows["total_tss"] > 30].copy()
    wp["month_dt"] = pd.to_datetime(wp["period"].astype(str))
    wp = wp.sort_values("month_dt").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(22, 12),
                              gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Training Composition vs FTP Progress  |  2019–2026",
                 fontsize=14, fontweight="bold", color=P["accent"])

    # ── Panel 1: stacked % TSS ────────────────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(wp))
    bottom = np.zeros(len(wp))

    for t in MAIN_TYPES:
        col  = _pct_col(t)
        vals = wp[col].fillna(0).values if col in wp.columns else np.zeros(len(wp))
        ax.bar(x, vals, bottom=bottom,
               color=TYPE_COLOURS.get(t, P["muted"]),
               label=t, alpha=0.85, width=0.9)
        bottom += vals

    # x-axis: label every 6 months
    tick_idx  = [i for i, m in enumerate(wp["month_dt"]) if m.month in (1, 7)]
    tick_lbls = [wp["month_dt"].iloc[i].strftime("%Y-%m") for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_lbls, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of monthly TSS", color=P["text"])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=7.5, ncol=6,
              bbox_to_anchor=(0, 1.13), framealpha=0.85)
    ax.set_title("Monthly Training Composition (% TSS by Type)", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # ── Panel 2: FTP proxy ────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(monthly_ftp["month_dt"], monthly_ftp["ftp_est"],
             color=P["purple"], lw=2.5, label="FTP proxy (best NP × 0.95)")
    ax2.fill_between(monthly_ftp["month_dt"], monthly_ftp["ftp_est"],
                     alpha=0.12, color=P["purple"])
    ax2.plot(monthly_ftp["month_dt"], monthly_ftp["ftp_trend"],
             color=P["accent"], lw=1.5, ls="--", label="3-month trend")
    ax2.axhline(275, color=P["green"],  lw=1,   ls="--", alpha=0.5, label="Peak 275W")
    ax2.axhline(235, color=P["yellow"], lw=1,   ls="--", alpha=0.7, label="Current 235W")
    ax2.axvline(pd.Timestamp("2025-06-01"), color=P["red"], lw=1.5, ls="--",
                alpha=0.7, label="Surgery Jun 2025")
    ax2.set_ylabel("Estimated FTP (W)", color=P["text"])
    ax2.legend(fontsize=7.5, loc="lower right", ncol=3)
    ax2.set_title("FTP Proxy — align peaks with composition periods above",
                  fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out = Path(save_dir) / "composition_analysis.png"
    plt.savefig(out, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  ✅ Composition chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame, save_dir: str = "outputs") -> dict:
    """Run the full composition analysis. Returns a dict of results."""

    calendar_windows = build_calendar_windows(df)
    rolling_windows  = build_rolling_windows(df, window_days=28, step_days=7)

    if len(calendar_windows) == 0:
        print("  ⚠️  No MAIN_TYPE sessions found — composition analysis skipped")
        return {}

    calendar_windows = classify_windows(calendar_windows)

    corr       = composition_correlations(calendar_windows)
    rf_results = run_composition_rf(calendar_windows)
    combos     = top_combinations(calendar_windows)
    mixed_summ = mixed_vs_single_summary(calendar_windows)

    print_summary(calendar_windows, corr, rf_results, combos, mixed_summ)

    Path(save_dir).mkdir(exist_ok=True)
    plot_composition(calendar_windows, df, save_dir=save_dir)

    return {
        "calendar_windows": calendar_windows,
        "rolling_windows":  rolling_windows,
        "correlations":     corr,
        "rf":               rf_results,
        "top_combinations": combos,
        "mixed_vs_single":  mixed_summ,
    }


if __name__ == "__main__":
    from src.data_loader import load
    df = load()
    results = run_all(df)
