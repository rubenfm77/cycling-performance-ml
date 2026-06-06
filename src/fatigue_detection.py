# src/fatigue_detection.py
"""
Fatigue anomaly detection — Isolation Forest.
Distinguishes genuine overreach from normal training load.
Flags sessions where multiple markers deviate simultaneously.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import PALETTE, PLOT_STYLE, ATHLETE


# Features that signal fatigue when they deviate together
FATIGUE_FEATURES = [
    "tss",           # high = stress
    "atl",           # high = accumulated fatigue
    "tsb",           # negative = fatigued
    "ramp_rate",     # >1.5 = ramping too fast
    "efficiency",    # dropping = cardiac fatigue signal
    "if_score",      # unexpectedly low for session type
    "hr_avg",        # high relative to power = cardiac strain
]

FATIGUE_LABELS = {
    "tss":        "TSS (Session Load)",
    "atl":        "ATL (7d Fatigue)",
    "tsb":        "TSB (Form)",
    "ramp_rate":  "Ramp Rate (ATL/CTL)",
    "efficiency": "Cardiac Efficiency (W/BPM)",
    "if_score":   "Intensity Factor",
    "hr_avg":     "Avg HR (bpm)",
}


def run_isolation_forest(df: pd.DataFrame,
                          contamination: float = 0.05) -> pd.DataFrame:
    """
    Isolation Forest anomaly detection.
    contamination = expected fraction of anomalies (5% default).
    Returns df with anomaly scores and flags.
    """
    df_clean = df.dropna(subset=FATIGUE_FEATURES).copy()

    X = df_clean[FATIGUE_FEATURES].values
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    df_clean["anomaly_score"] = iso.fit_predict(X_scaled)
    df_clean["anomaly_raw"]   = iso.score_samples(X_scaled)
    # -1 = anomaly, +1 = normal → convert to bool
    df_clean["is_anomaly"]    = df_clean["anomaly_score"] == -1

    # Severity: how extreme is the anomaly score
    df_clean["anomaly_severity"] = np.where(
        df_clean["is_anomaly"],
        (-df_clean["anomaly_raw"]).clip(lower=0),
        0
    )

    # Context label for each anomaly
    def label_anomaly(row):
        if not row["is_anomaly"]:
            return "Normal"
        flags = []
        if row["tsb"] < -30:      flags.append("Deep fatigue (TSB<-30)")
        if row["ramp_rate"] > 1.5: flags.append("Ramp too fast")
        if row["tss"] > df_clean["tss"].quantile(0.95): flags.append("Extreme TSS")
        if row["efficiency"] < df_clean["efficiency"].quantile(0.1): flags.append("Low efficiency")
        return " + ".join(flags) if flags else "Multi-factor anomaly"

    df_clean["anomaly_label"] = df_clean.apply(label_anomaly, axis=1)

    n_anomalies = df_clean["is_anomaly"].sum()
    print(f"\n{'='*55}")
    print(f"  FATIGUE ANOMALY DETECTION")
    print(f"{'='*55}")
    print(f"  Total sessions analysed : {len(df_clean)}")
    print(f"  Anomalies detected      : {n_anomalies} ({n_anomalies/len(df_clean)*100:.1f}%)")
    print(f"\n  Anomalous sessions:")
    anomalies = df_clean[df_clean["is_anomaly"]].sort_values("anomaly_severity", ascending=False)
    for _, row in anomalies.head(15).iterrows():
        print(f"    {row['date'].date()}  {row['training_type']:<18} "
              f"TSB={row['tsb']:+.0f}  TSS={row['tss']:.0f}  "
              f"Ramp={row['ramp_rate']:.2f}  → {row['anomaly_label']}")

    return df_clean


def plot_fatigue_detection(df_anomaly: pd.DataFrame,
                            save_dir: str = "outputs") -> None:
    """4-panel fatigue anomaly visualisation."""

    plt.rcParams.update(PLOT_STYLE)
    P    = PALETTE
    surgery = pd.Timestamp(ATHLETE["surgery_date"])

    normal   = df_anomaly[~df_anomaly["is_anomaly"]]
    anomalies = df_anomaly[df_anomaly["is_anomaly"]]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Fatigue Anomaly Detection  |  Isolation Forest",
                 fontsize=14, fontweight="bold", color=P["accent"])

    # ── 1. TSB over time with anomalies flagged ───────────────────────────────
    ax = axes[0, 0]
    ax.plot(df_anomaly["date"], df_anomaly["tsb"],
            color=P["accent"], lw=1.2, alpha=0.5, label="TSB")
    ax.scatter(normal["date"],   normal["tsb"],
               color=P["green"], s=8, alpha=0.4, label="Normal")
    ax.scatter(anomalies["date"], anomalies["tsb"],
               color=P["red"], s=60, alpha=0.9, zorder=5,
               marker="X", label=f"Anomaly (n={len(anomalies)})")
    ax.axhline(-30, color=P["red"],    lw=1, ls="--", alpha=0.5)
    ax.axhline(25,  color=P["purple"], lw=1, ls="--", alpha=0.5)
    ax.axvline(surgery, color=P["orange"], lw=1.5, ls="--", alpha=0.7)
    ax.set_ylabel("TSB (Form)", color=P["text"])
    ax.set_title("TSB with Anomaly Flags", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # ── 2. Anomaly score over time ────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.plot(df_anomaly["date"], df_anomaly["anomaly_raw"],
             color=P["muted"], lw=1, alpha=0.6)
    ax2.scatter(anomalies["date"], anomalies["anomaly_raw"],
                color=P["red"], s=50, zorder=5, alpha=0.9,
                marker="X", label="Anomaly")
    threshold_line = df_anomaly.loc[df_anomaly["is_anomaly"], "anomaly_raw"].max()
    ax2.axhline(threshold_line, color=P["red"], lw=1, ls="--",
                label=f"Anomaly threshold ({threshold_line:.3f})")
    ax2.axvline(surgery, color=P["orange"], lw=1.5, ls="--", alpha=0.7,
                label="Surgery")
    ax2.set_ylabel("Isolation Forest Score\n(lower = more anomalous)")
    ax2.set_title("Anomaly Score Timeline", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── 3. Ramp rate with anomalies ───────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.plot(df_anomaly["date"], df_anomaly["ramp_rate"],
             color=P["accent"], lw=1.2, alpha=0.5)
    ax3.scatter(anomalies["date"], anomalies["ramp_rate"],
                color=P["red"], s=50, zorder=5, alpha=0.9, marker="X")
    ax3.axhline(1.5, color=P["red"],    lw=1.5, ls="--", label="Injury risk (1.5)")
    ax3.axhline(1.2, color=P["orange"], lw=1,   ls="--", label="High ramp (1.2)")
    ax3.axhline(1.0, color=P["green"],  lw=1,   ls="--", label="Balanced (1.0)")
    ax3.set_ylabel("Ramp Rate (ATL/CTL)")
    ax3.set_title("Ramp Rate — Overreach Risk", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── 4. Feature distribution: anomalies vs normal ──────────────────────────
    ax4 = axes[1, 1]
    features_to_compare = ["tsb", "ramp_rate", "efficiency", "tss"]
    labels_compare = ["TSB", "Ramp Rate", "W/BPM", "TSS"]
    x = np.arange(len(features_to_compare))
    width = 0.35

    # Normalise for comparison
    for i, (feat, label) in enumerate(zip(features_to_compare, labels_compare)):
        all_vals  = df_anomaly[feat].dropna()
        norm_mean = all_vals.mean()
        norm_std  = all_vals.std() + 1e-9
        n_mean = (normal[feat].mean() - norm_mean) / norm_std
        a_mean = (anomalies[feat].mean() - norm_mean) / norm_std
        ax4.bar(i - width/2, n_mean, width,
                color=P["green"], alpha=0.75,
                label="Normal" if i == 0 else "")
        ax4.bar(i + width/2, a_mean, width,
                color=P["red"], alpha=0.75,
                label="Anomaly" if i == 0 else "")
    ax4.axhline(0, color=P["muted"], lw=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_compare)
    ax4.set_ylabel("Z-score vs population mean")
    ax4.set_title("Anomaly vs Normal Session Profile\n(normalised)", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = Path(save_dir) / "fatigue_detection.png"
    plt.savefig(out, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  ✅ Fatigue detection chart saved → {out}")


if __name__ == "__main__":
    from src.data_loader import load
    df = load()
    df_anomaly = run_isolation_forest(df)
    plot_fatigue_detection(df_anomaly)
