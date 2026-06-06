# src/ftp_analysis.py
"""
ML analysis: which training types and session features best drive FTP gains.

Models:
  1. Random Forest classifier — predict high vs low FTP stimulus sessions
  2. Feature importance (MDI + permutation) — what actually matters
  3. Per-type FTP stimulus scoring and ranking
  4. Correlation analysis between session features and FTP proxy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from scipy.stats import pearsonr, spearmanr

from src.config import PALETTE, PLOT_STYLE, MAIN_TYPES, TYPE_COLOURS


# ── Feature set for ML ────────────────────────────────────────────────────────
FEATURES = [
    "if_score", "duration_h", "tss", "power_avg", "w_per_kg",
    "z2_pct", "z3_pct", "z4_pct", "z5_pct", "z4plus_pct",
    "elevation", "elev_per_km", "temp_avg", "cadence",
    "ctl", "tsb", "ramp_rate",
]

FEATURE_LABELS = {
    "if_score":      "Intensity Factor (IF)",
    "duration_h":    "Duration (hours)",
    "tss":           "TSS",
    "power_avg":     "Avg Power (W)",
    "w_per_kg":      "W/kg",
    "z2_pct":        "% Time Z2 (Endurance)",
    "z3_pct":        "% Time Z3 (Tempo)",
    "z4_pct":        "% Time Z4 (Threshold)",
    "z5_pct":        "% Time Z5 (VO2max)",
    "z4plus_pct":    "% Time Z4+ (High Intensity)",
    "elevation":     "Elevation Gain (m)",
    "elev_per_km":   "Elevation per km",
    "temp_avg":      "Temperature (°C)",
    "cadence":       "Avg Cadence (rpm)",
    "ctl":           "CTL (Fitness)",
    "tsb":           "TSB (Form)",
    "ramp_rate":     "Ramp Rate (ATL/CTL)",
}


def _prepare_data(df: pd.DataFrame):
    """Prepare feature matrix for ML. Returns X, y_stimulus, y_type, df_clean."""
    df_ml = df[df["training_type"].isin(MAIN_TYPES)].copy()
    df_ml = df_ml.dropna(subset=FEATURES + ["ftp_stimulus"])

    X = df_ml[FEATURES].copy()

    # Target 1: FTP stimulus score (continuous)
    y_stim = df_ml["ftp_stimulus"].values

    # Target 2: Binary — high FTP stimulus session (top tercile)
    threshold = np.percentile(y_stim, 67)
    y_binary = (y_stim >= threshold).astype(int)

    # Target 3: Training type
    le = LabelEncoder()
    y_type = le.fit_transform(df_ml["training_type"])

    return X, y_stim, y_binary, y_type, le, df_ml


def run_random_forest(df: pd.DataFrame) -> dict:
    """
    Train Random Forest to predict high-FTP-stimulus sessions.
    Returns feature importances and CV scores.
    """
    X, y_stim, y_binary, _, _, df_ml = _prepare_data(df)

    # ── Classifier: high vs low FTP stimulus ─────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y_binary, cv=cv, scoring="roc_auc")

    clf.fit(X, y_binary)

    # MDI importance
    mdi_imp = pd.Series(
        clf.feature_importances_,
        index=[FEATURE_LABELS.get(f, f) for f in FEATURES]
    ).sort_values(ascending=False)

    # Permutation importance (more reliable)
    perm = permutation_importance(clf, X, y_binary,
                                  n_repeats=20, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(
        perm.importances_mean,
        index=[FEATURE_LABELS.get(f, f) for f in FEATURES]
    ).sort_values(ascending=False)

    # ── Regressor: predict FTP stimulus score directly ────────────────────────
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    reg_scores = cross_val_score(reg, X, y_stim, cv=5, scoring="r2")

    print(f"\n{'='*55}")
    print("  RANDOM FOREST — FTP STIMULUS ANALYSIS")
    print(f"{'='*55}")
    print(f"  Classifier ROC-AUC  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Regressor R²        : {reg_scores.mean():.3f} ± {reg_scores.std():.3f}")
    print(f"\n  Top features (MDI importance):")
    for feat, imp in mdi_imp.head(8).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat:<35} {imp:.3f}  {bar}")

    return {
        "classifier":   clf,
        "mdi_imp":      mdi_imp,
        "perm_imp":     perm_imp,
        "cv_auc":       cv_scores,
        "reg_r2":       reg_scores,
        "X":            X,
        "y_binary":     y_binary,
        "df_ml":        df_ml,
    }


def type_stimulus_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank training types by FTP stimulus score."""
    df_main = df[df["training_type"].isin(MAIN_TYPES)].copy()

    ranking = (
        df_main.groupby("training_type")
        .agg(
            n            = ("tss", "count"),
            avg_stimulus = ("ftp_stimulus", "mean"),
            avg_if       = ("if_score", "mean"),
            avg_hours    = ("duration_h", "mean"),
            avg_tss      = ("tss", "mean"),
            avg_z4pct    = ("z4_pct", "mean"),
            avg_z5pct    = ("z5_pct", "mean"),
        )
        .sort_values("avg_stimulus", ascending=False)
        .reset_index()
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking["high_intensity_total_pct"] = ranking["avg_z4pct"] + ranking["avg_z5pct"]

    print(f"\n{'='*70}")
    print("  FTP STIMULUS RANKING BY TRAINING TYPE")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Type':<20} {'Score':>7} {'IF':>6} {'Hours':>6} {'Z4%':>6} {'n':>5}")
    print(f"  {'-'*60}")
    for _, r in ranking.iterrows():
        print(f"  {r['rank']:<5} {r['training_type']:<20} "
              f"{r['avg_stimulus']:>7.1f} {r['avg_if']:>6.3f} "
              f"{r['avg_hours']:>6.2f} {r['avg_z4pct']:>6.1f} {r['n']:>5}")

    return ranking


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson + Spearman correlations between all features and FTP stimulus."""
    df_clean = df[df["training_type"].isin(MAIN_TYPES)].copy()
    df_clean = df_clean.dropna(subset=FEATURES + ["ftp_stimulus"])

    results = []
    for feat in FEATURES:
        r_p, p_p = pearsonr(df_clean[feat], df_clean["ftp_stimulus"])
        r_s, p_s = spearmanr(df_clean[feat], df_clean["ftp_stimulus"])
        results.append({
            "feature":    FEATURE_LABELS.get(feat, feat),
            "pearson_r":  round(r_p, 3),
            "pearson_p":  round(p_p, 4),
            "spearman_r": round(r_s, 3),
            "spearman_p": round(p_s, 4),
            "sig":        "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns",
        })

    return pd.DataFrame(results).sort_values("pearson_r", key=abs, ascending=False)


def plot_ftp_analysis(df: pd.DataFrame, rf_results: dict,
                      ranking: pd.DataFrame, save_dir: str = "outputs") -> None:
    """4-panel FTP analysis figure."""

    plt.rcParams.update(PLOT_STYLE)
    P = PALETTE

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("FTP Development Analysis  |  What drives threshold adaptation?",
                 fontsize=14, fontweight="bold", color=P["accent"])

    # ── 1. FTP Stimulus Ranking ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    colors = [
        P["green"] if s >= 50 else P["accent"] if s >= 35 else P["yellow"]
        for s in ranking["avg_stimulus"]
    ]
    bars = ax1.barh(ranking["training_type"][::-1],
                    ranking["avg_stimulus"][::-1],
                    color=colors[::-1], alpha=0.85)
    ax1.set_xlabel("FTP Stimulus Score  (IF² × Duration × 100)")
    ax1.set_title("FTP Stimulus Score by Training Type", fontweight="bold")
    ax1.xaxis.grid(True, alpha=0.3)
    for bar, row in zip(bars, ranking.iloc[::-1].itertuples()):
        ax1.text(bar.get_width() + 0.5,
                 bar.get_y() + bar.get_height() / 2,
                 f"n={row.n}  IF={row.avg_if:.2f}",
                 va="center", fontsize=8, color=P["muted"])

    # ── 2. Feature importance (MDI) ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    top8 = rf_results["mdi_imp"].head(8)
    bars2 = ax2.barh(top8.index[::-1], top8.values[::-1],
                     color=P["purple"], alpha=0.85)
    ax2.set_xlabel("MDI Importance")
    ax2.set_title(f"RF Feature Importance\nROC-AUC={rf_results['cv_auc'].mean():.3f}",
                  fontweight="bold")
    ax2.xaxis.grid(True, alpha=0.3)

    # ── 3. IF distribution per type (violin) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    types_sorted = ranking["training_type"].tolist()
    data_violin  = [
        df[df["training_type"] == t]["if_score"].dropna().values
        for t in types_sorted
    ]
    data_violin_clean = [(t, d) for t, d in zip(types_sorted, data_violin) if len(d) > 3]
    labels_v = [t for t, _ in data_violin_clean]
    data_v   = [d for _, d in data_violin_clean]

    vp = ax3.violinplot(data_v, vert=False, showmedians=True, showextrema=False)
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(TYPE_COLOURS.get(labels_v[i], P["accent"]))
        pc.set_alpha(0.75)
    vp["cmedians"].set_color("white")
    ax3.set_yticks(range(1, len(labels_v) + 1))
    ax3.set_yticklabels(labels_v, fontsize=9)
    ax3.set_xlabel("Intensity Factor (IF)")
    ax3.axvline(0.75, color=P["orange"], lw=1, ls="--", label="SST threshold (0.75)")
    ax3.axvline(0.85, color=P["red"],    lw=1, ls="--", label="FTP threshold (0.85)")
    ax3.legend(fontsize=8)
    ax3.set_title("IF Distribution per Training Type", fontweight="bold")
    ax3.xaxis.grid(True, alpha=0.3)

    # ── 4. Z4 time vs FTP stimulus scatter ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    df_main = df[df["training_type"].isin(MAIN_TYPES)].dropna(subset=["z4_pct", "ftp_stimulus"])
    for ttype in types_sorted:
        sub = df_main[df_main["training_type"] == ttype]
        ax4.scatter(sub["z4_pct"], sub["ftp_stimulus"],
                    color=TYPE_COLOURS.get(ttype, P["accent"]),
                    alpha=0.5, s=20, label=ttype)
    ax4.set_xlabel("% Time in Z4 (Threshold)")
    ax4.set_ylabel("FTP Stimulus Score")
    ax4.set_title("Z4 Time % vs FTP Stimulus", fontweight="bold")
    ax4.legend(fontsize=6.5, ncol=2)
    ax4.grid(alpha=0.2)

    plt.savefig(Path(save_dir) / "ftp_analysis.png",
                bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print("  ✅ FTP analysis chart saved → outputs/ftp_analysis.png")


if __name__ == "__main__":
    from src.data_loader import load
    df = load()
    rf   = run_random_forest(df)
    rank = type_stimulus_ranking(df)
    corr = correlation_analysis(df)
    print("\nTop correlations with FTP stimulus:")
    print(corr.head(8).to_string(index=False))
    plot_ftp_analysis(df, rf, rank)
