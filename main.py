# main.py
"""
Cycling Performance ML — Full Pipeline Runner
=============================================
Run all analyses in sequence and save outputs.

Usage:
    python main.py                    # full pipeline
    python main.py --module ftp       # single module
    python main.py --module pmc
    python main.py --module wkg
    python main.py --module cluster
    python main.py --module fatigue
"""

import argparse
import time
from pathlib import Path


def run_pipeline(module: str = "all") -> None:

    start = time.time()
    print("\n" + "🚴 " * 20)
    print("  CYCLING PERFORMANCE ML PIPELINE")
    print("🚴 " * 20)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n📥 Loading data...")
    from src.data_loader import load
    df = load()

    Path("outputs").mkdir(exist_ok=True)

    # ── PMC ───────────────────────────────────────────────────────────────────
    if module in ("all", "pmc"):
        print("\n📈 Running Performance Management Chart...")
        from src.pmc import plot_pmc, fatigue_summary, ramp_rate_analysis
        plot_pmc(df)
        print("\n  Fatigue state distribution:")
        print(fatigue_summary(df).to_string(index=False))

    # ── FTP Analysis ──────────────────────────────────────────────────────────
    if module in ("all", "ftp"):
        print("\n🎯 Running FTP Analysis (Random Forest)...")
        from src.ftp_analysis import (run_random_forest, type_stimulus_ranking,
                                       correlation_analysis, plot_ftp_analysis)
        rf_results = run_random_forest(df)
        ranking    = type_stimulus_ranking(df)
        corr       = correlation_analysis(df)
        plot_ftp_analysis(df, rf_results, ranking)
        print("\n  Top correlations with FTP stimulus:")
        print(corr.head(6)[["feature","pearson_r","sig"]].to_string(index=False))

    # ── W/kg Progression ──────────────────────────────────────────────────────
    if module in ("all", "wkg"):
        print("\n⚡ Running W/kg Progression Analysis...")
        from src.wkg_progression import plot_wkg_progression, forecast_wkg
        plot_wkg_progression(df)
        forecast_wkg(df, weeks_ahead=12)

    # ── Clustering ────────────────────────────────────────────────────────────
    if module in ("all", "cluster"):
        print("\n🔬 Running Session Clustering (K-Means + PCA)...")
        from src.clustering import run_clustering, plot_clustering
        df_clustered, km, pca, scaler, k_results = run_clustering(df)
        plot_clustering(df_clustered, k_results)

    # ── Fatigue Detection ─────────────────────────────────────────────────────
    if module in ("all", "fatigue"):
        print("\n😴 Running Fatigue Anomaly Detection (Isolation Forest)...")
        from src.fatigue_detection import run_isolation_forest, plot_fatigue_detection
        df_anomaly = run_isolation_forest(df)
        plot_fatigue_detection(df_anomaly)

    # ── Composition Analysis ──────────────────────────────────────────────────
    if module in ("all", "composition"):
        print("\n🔀 Running Training Composition Analysis...")
        from src.monthly_composition_analysis import run_all as run_composition
        run_composition(df)

    elapsed = time.time() - start
    print(f"\n{'='*55}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"  📁 Outputs saved to: outputs/")
    print(f"     pmc_full.png")
    print(f"     ftp_analysis.png")
    print(f"     wkg_progression.png")
    print(f"     clustering.png")
    print(f"     fatigue_detection.png")
    print(f"     composition_analysis.png")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cycling Performance ML Pipeline")
    parser.add_argument("--module", default="all",
                        choices=["all","pmc","ftp","wkg","cluster","fatigue","composition"],
                        help="Which module to run (default: all)")
    args = parser.parse_args()
    run_pipeline(args.module)
