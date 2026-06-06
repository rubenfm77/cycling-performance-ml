# src/clustering.py
"""
Session clustering — K-Means + PCA.
Finds natural archetypes in the training data beyond the labelled types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

from src.config import PALETTE, PLOT_STYLE, MAIN_TYPES, TYPE_COLOURS


CLUSTER_FEATURES = [
    "if_score", "duration_h", "tss", "w_per_kg",
    "z2_pct", "z3_pct", "z4_pct", "z5_pct",
    "elevation", "elev_per_km", "hr_avg", "efficiency",
]

CLUSTER_LABELS = {
    "if_score":    "IF",
    "duration_h":  "Duration (h)",
    "tss":         "TSS",
    "w_per_kg":    "W/kg",
    "z2_pct":      "Z2 %",
    "z3_pct":      "Z3 %",
    "z4_pct":      "Z4 %",
    "z5_pct":      "Z5 %",
    "elevation":   "Elevation (m)",
    "elev_per_km": "Elev/km",
    "hr_avg":      "Avg HR",
    "efficiency":  "W/BPM",
}

# Human-readable cluster archetype names (assigned after inspection)
ARCHETYPE_NAMES = {
    0: "🟢 Long Endurance",
    1: "🔵 Aerobic Base",
    2: "🟡 Tempo/SST",
    3: "🔴 Threshold",
    4: "🟣 VO2max / Race",
    5: "⚪ Recovery",
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> dict:
    """Silhouette + Davies-Bouldin to find optimal number of clusters."""
    sil_scores = []
    db_scores  = []
    inertias   = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        inertias.append(km.inertia_)

    best_k_sil = k_range[np.argmax(sil_scores)]
    best_k_db  = k_range[np.argmin(db_scores)]

    print(f"\n  Optimal K (Silhouette) : {best_k_sil}  (score={max(sil_scores):.3f})")
    print(f"  Optimal K (Davies-Bouldin) : {best_k_db}")

    return {
        "k_range":    list(k_range),
        "sil_scores": sil_scores,
        "db_scores":  db_scores,
        "inertias":   inertias,
        "best_k":     best_k_sil,
    }


def run_clustering(df: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    """Run K-Means + PCA. Returns df with cluster labels and PCA coordinates."""

    df_clean = df.dropna(subset=CLUSTER_FEATURES).copy()
    X = df_clean[CLUSTER_FEATURES].values

    # Scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal K first
    k_results = find_optimal_k(X_scaled)
    best_k    = k_results["best_k"]
    if n_clusters != best_k:
        print(f"  ℹ️  Using n_clusters={n_clusters} (optimal={best_k})")

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
    df_clean["cluster"] = km.fit_predict(X_scaled)

    # PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_clean["pca_x"] = coords[:, 0]
    df_clean["pca_y"] = coords[:, 1]
    df_clean["pca_var_1"] = pca.explained_variance_ratio_[0]
    df_clean["pca_var_2"] = pca.explained_variance_ratio_[1]

    total_var = pca.explained_variance_ratio_.sum()
    print(f"\n  PCA variance explained (2 components): {total_var:.1%}")

    # Cluster profiles
    profile = (
        df_clean.groupby("cluster")[CLUSTER_FEATURES]
        .mean()
        .round(2)
    )

    print(f"\n{'='*65}")
    print("  CLUSTER PROFILES")
    print(f"{'='*65}")
    print(profile.to_string())

    # Cluster size and dominant training type
    print(f"\n  Cluster composition:")
    for c in range(n_clusters):
        sub = df_clean[df_clean["cluster"] == c]
        top_type = sub["training_type"].value_counts().index[0] if len(sub) > 0 else "?"
        archetype = ARCHETYPE_NAMES.get(c, f"Cluster {c}")
        print(f"    {archetype:<25} n={len(sub):>4}  "
              f"dominant type: {top_type:<18} "
              f"avg IF={sub['if_score'].mean():.2f}  "
              f"avg TSS={sub['tss'].mean():.0f}")

    return df_clean, km, pca, scaler, k_results


def plot_clustering(df_clustered: pd.DataFrame, k_results: dict,
                    save_dir: str = "outputs") -> None:
    """4-panel clustering visualisation."""

    plt.rcParams.update(PLOT_STYLE)
    P = PALETTE

    n_clusters = df_clustered["cluster"].nunique()
    cluster_colors = [
        P["green"], P["accent"], P["yellow"],
        P["red"], P["purple"], P["muted"],
    ][:n_clusters]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Session Clustering  |  K={n_clusters}  ·  K-Means + PCA",
                 fontsize=14, fontweight="bold", color=P["accent"])

    # ── 1. PCA scatter coloured by cluster ───────────────────────────────────
    ax = axes[0, 0]
    for c in range(n_clusters):
        sub = df_clustered[df_clustered["cluster"] == c]
        ax.scatter(sub["pca_x"], sub["pca_y"],
                   color=cluster_colors[c], alpha=0.6, s=20,
                   label=ARCHETYPE_NAMES.get(c, f"Cluster {c}"))
    var1 = df_clustered["pca_var_1"].iloc[0] * 100
    var2 = df_clustered["pca_var_2"].iloc[0] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title("PCA — Cluster Separation", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ── 2. Silhouette score by K ──────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.plot(k_results["k_range"], k_results["sil_scores"],
             color=P["accent"], lw=2, marker="o", ms=8, label="Silhouette")
    ax2_r = ax2.twinx()
    ax2_r.plot(k_results["k_range"], k_results["db_scores"],
               color=P["orange"], lw=2, marker="s", ms=6, ls="--",
               label="Davies-Bouldin (lower=better)")
    ax2_r.set_ylabel("Davies-Bouldin Score", color=P["orange"])
    ax2_r.tick_params(axis="y", labelcolor=P["orange"])
    best_k = k_results["best_k"]
    ax2.axvline(best_k, color=P["green"], lw=1.5, ls="--",
                label=f"Optimal K={best_k}")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score", color=P["accent"])
    ax2.set_title("Optimal K Selection", fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax2.grid(alpha=0.2)

    # ── 3. IF vs Duration scatter coloured by cluster ─────────────────────────
    ax3 = axes[1, 0]
    for c in range(n_clusters):
        sub = df_clustered[df_clustered["cluster"] == c]
        ax3.scatter(sub["duration_h"], sub["if_score"],
                    color=cluster_colors[c], alpha=0.6, s=20,
                    label=ARCHETYPE_NAMES.get(c, f"Cluster {c}"))
    ax3.axhline(0.75, color=P["orange"], lw=1, ls="--", alpha=0.5)
    ax3.axhline(0.85, color=P["red"],    lw=1, ls="--", alpha=0.5)
    ax3.set_xlabel("Duration (hours)")
    ax3.set_ylabel("Intensity Factor (IF)")
    ax3.set_title("IF vs Duration by Cluster", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)

    # ── 4. Cluster profile radar / bar ────────────────────────────────────────
    ax4 = axes[1, 1]
    profile_norm = (
        df_clustered.groupby("cluster")[
            ["if_score", "duration_h", "z4_pct", "z2_pct", "tss", "w_per_kg"]
        ]
        .mean()
    )
    # Normalize 0-1 for comparison
    profile_norm = (profile_norm - profile_norm.min()) / \
                   (profile_norm.max() - profile_norm.min() + 1e-9)

    x = np.arange(len(profile_norm.columns))
    width = 0.8 / n_clusters
    for i, (idx, row) in enumerate(profile_norm.iterrows()):
        ax4.bar(x + i * width, row.values,
                width, color=cluster_colors[i], alpha=0.75,
                label=ARCHETYPE_NAMES.get(idx, f"Cluster {idx}"))
    ax4.set_xticks(x + width * (n_clusters - 1) / 2)
    ax4.set_xticklabels(["IF", "Duration", "Z4%", "Z2%", "TSS", "W/kg"],
                         fontsize=9)
    ax4.set_ylabel("Normalised value (0–1)")
    ax4.set_title("Cluster Profile Comparison (normalised)", fontweight="bold")
    ax4.legend(fontsize=7.5, ncol=2)
    ax4.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = Path(save_dir) / "clustering.png"
    plt.savefig(out, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  ✅ Clustering chart saved → {out}")


if __name__ == "__main__":
    from src.data_loader import load
    df = load()
    print("\n" + "="*55)
    print("  SESSION CLUSTERING")
    print("="*55)
    df_clustered, km, pca, scaler, k_results = run_clustering(df)
    plot_clustering(df_clustered, k_results)
