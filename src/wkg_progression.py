# src/wkg_progression.py
"""
W/kg progression analysis.
Linear regression + Ridge + structural break detection (surgery Jun 2025).
Forecasts W/kg trajectory for next 12 weeks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

from src.config import PALETTE, PLOT_STYLE, ATHLETE


def _monthly_wkg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate W/kg by month with confidence interval proxy."""
    monthly = (
        df[df["w_per_kg"] > 0]
        .groupby(df["month"])
        .agg(
            avg_wkg   = ("w_per_kg", "mean"),
            std_wkg   = ("w_per_kg", "std"),
            max_wkg   = ("w_per_kg", "max"),
            n         = ("w_per_kg", "count"),
            avg_power = ("power_avg", "mean"),
            avg_tss   = ("tss", "mean"),
        )
        .reset_index()
    )
    monthly["month_dt"] = monthly["month"].apply(lambda p: p.start_time)
    monthly["month_num"] = np.arange(len(monthly))
    monthly["ci_upper"]  = monthly["avg_wkg"] + monthly["std_wkg"].fillna(0)
    monthly["ci_lower"]  = monthly["avg_wkg"] - monthly["std_wkg"].fillna(0)
    return monthly


def fit_progression_models(df: pd.DataFrame) -> dict:
    """Fit pre-surgery and post-surgery regression models."""
    surgery = pd.Timestamp(ATHLETE["surgery_date"])
    monthly = _monthly_wkg(df)

    pre  = monthly[monthly["month_dt"] <  surgery].dropna(subset=["avg_wkg"])
    post = monthly[monthly["month_dt"] >= surgery].dropna(subset=["avg_wkg"])

    results = {}

    for label, subset in [("pre_surgery", pre), ("post_surgery", post)]:
        if len(subset) < 3:
            continue
        X = subset["month_num"].values.reshape(-1, 1)
        y = subset["avg_wkg"].values

        # Linear
        lin = LinearRegression().fit(X, y)
        lin_r2 = r2_score(y, lin.predict(X))

        # Polynomial degree 2
        poly_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0))
        ])
        if len(subset) >= 5:
            poly_pipe.fit(X, y)
            poly_r2 = r2_score(y, poly_pipe.predict(X))
        else:
            poly_pipe = lin
            poly_r2 = lin_r2

        results[label] = {
            "model_lin":  lin,
            "model_poly": poly_pipe,
            "r2_lin":     lin_r2,
            "r2_poly":    poly_r2,
            "slope_wkg_per_month": lin.coef_[0],
            "X": X, "y": y,
            "subset": subset,
        }

        direction = "📈" if lin.coef_[0] > 0 else "📉"
        print(f"\n  {label.upper().replace('_',' ')}:")
        print(f"    Months of data : {len(subset)}")
        print(f"    Linear R²      : {lin_r2:.3f}")
        print(f"    Slope          : {lin.coef_[0]:+.4f} W/kg per month")
        print(f"    Trend          : {direction} {abs(lin.coef_[0]*12):.2f} W/kg per year")

    return results, monthly


def forecast_wkg(df: pd.DataFrame, weeks_ahead: int = 12) -> pd.DataFrame:
    """
    Forecast W/kg for next N weeks using post-surgery trend.
    Returns DataFrame with date, predicted W/kg, and confidence bounds.
    """
    results, monthly = fit_progression_models(df)

    if "post_surgery" not in results:
        print("  ⚠️  Not enough post-surgery data for forecast")
        return pd.DataFrame()

    post_model = results["post_surgery"]
    last_month_num = monthly["month_num"].max()
    last_date      = monthly["month_dt"].max()

    future_months = np.arange(last_month_num + 1,
                               last_month_num + 1 + weeks_ahead // 4 + 1)
    X_future = future_months.reshape(-1, 1)
    y_pred   = post_model["model_lin"].predict(X_future)

    future_dates = [last_date + pd.DateOffset(months=i+1)
                    for i in range(len(future_months))]

    # Simple uncertainty: ±1 std of residuals
    residuals = post_model["y"] - post_model["model_lin"].predict(post_model["X"])
    std_res   = residuals.std()

    forecast = pd.DataFrame({
        "date":      future_dates,
        "wkg_pred":  y_pred,
        "wkg_upper": y_pred + 1.5 * std_res,
        "wkg_lower": y_pred - 1.5 * std_res,
    })

    print(f"\n  W/kg FORECAST (+{weeks_ahead} weeks):")
    for _, row in forecast.iterrows():
        print(f"    {row['date'].strftime('%b %Y')}: "
              f"{row['wkg_pred']:.2f} W/kg "
              f"[{row['wkg_lower']:.2f} – {row['wkg_upper']:.2f}]")

    return forecast


def plot_wkg_progression(df: pd.DataFrame, save_dir: str = "outputs") -> None:
    """3-panel W/kg progression chart."""

    plt.rcParams.update(PLOT_STYLE)
    P = PALETTE
    surgery = pd.Timestamp(ATHLETE["surgery_date"])
    results, monthly = fit_progression_models(df)
    forecast = forecast_wkg(df)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("W/kg Progression  |  2019–2026 + 12-week Forecast",
                 fontsize=14, fontweight="bold", color=P["accent"])

    # ── Panel 1: Monthly W/kg with trend lines ────────────────────────────────
    ax = axes[0]
    ax.fill_between(monthly["month_dt"],
                    monthly["ci_lower"], monthly["ci_upper"],
                    alpha=0.15, color=P["purple"])
    ax.plot(monthly["month_dt"], monthly["avg_wkg"],
            color=P["purple"], lw=2, marker="o", ms=4, label="Monthly avg W/kg")
    ax.plot(monthly["month_dt"], monthly["max_wkg"],
            color=P["accent"], lw=1, ls="--", alpha=0.6, label="Monthly max W/kg")

    # Pre-surgery trend
    if "pre_surgery" in results:
        r = results["pre_surgery"]
        x_line = r["subset"]["month_dt"].values
        ax.plot(x_line, r["model_lin"].predict(r["X"]),
                color=P["green"], lw=2, ls="-.", label=f"Pre-surgery trend (R²={r['r2_lin']:.2f})")

    # Post-surgery trend
    if "post_surgery" in results:
        r = results["post_surgery"]
        x_line = r["subset"]["month_dt"].values
        ax.plot(x_line, r["model_lin"].predict(r["X"]),
                color=P["orange"], lw=2, ls="-.", label=f"Post-surgery trend (R²={r['r2_lin']:.2f})")

    # Forecast
    if not forecast.empty:
        ax.fill_between(forecast["date"],
                        forecast["wkg_lower"], forecast["wkg_upper"],
                        alpha=0.2, color=P["green"])
        ax.plot(forecast["date"], forecast["wkg_pred"],
                color=P["green"], lw=2, ls="--", marker="s", ms=5, label="Forecast")

    # Surgery annotation
    ax.axvline(surgery, color=P["red"], lw=1.5, ls="--", alpha=0.7)
    ax.text(surgery, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 2.0,
            " ⚕ Surgery Jun 2025", color=P["red"], fontsize=9, va="bottom")

    # Target FTP line
    target_wkg = ATHLETE["ftp_target"] / ATHLETE["weight_kg"]
    ax.axhline(target_wkg, color=P["yellow"], lw=1, ls=":",
               label=f"Pre-accident FTP target ({target_wkg:.2f} W/kg)")

    ax.set_ylabel("W/kg", color=P["text"])
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Monthly W/kg — Trend, Surgery Break & Forecast", fontweight="bold")
    ax.grid(alpha=0.25)

    # ── Panel 2: Session W/kg coloured by training type ───────────────────────
    from src.config import TYPE_COLOURS, MAIN_TYPES
    ax2 = axes[1]
    df_plot = df[df["training_type"].isin(MAIN_TYPES) & (df["w_per_kg"] > 0)]
    for ttype in MAIN_TYPES:
        sub = df_plot[df_plot["training_type"] == ttype]
        ax2.scatter(sub["date"], sub["w_per_kg"],
                    color=TYPE_COLOURS.get(ttype, P["accent"]),
                    alpha=0.5, s=15, label=ttype)
    ax2.axvline(surgery, color=P["red"], lw=1.5, ls="--", alpha=0.7)
    ax2.set_ylabel("W/kg per Session", color=P["text"])
    ax2.legend(fontsize=7, ncol=3, loc="upper left")
    ax2.set_title("Session W/kg by Training Type", fontweight="bold")
    ax2.grid(alpha=0.2)

    # ── Panel 3: Yearly W/kg boxplot ─────────────────────────────────────────
    ax3 = axes[2]
    years = sorted(df["year"].unique())
    data_by_year = [df[df["year"] == y]["w_per_kg"].dropna().values for y in years]
    bp = ax3.boxplot(data_by_year, labels=years, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     whiskerprops=dict(color=P["muted"]),
                     capprops=dict(color=P["muted"]),
                     flierprops=dict(marker="o", color=P["muted"],
                                     alpha=0.3, ms=3))
    colors_yr = [P["accent"]] * len(years)
    # Highlight surgery year
    if 2025 in years:
        colors_yr[years.index(2025)] = P["red"]
    for patch, color in zip(bp["boxes"], colors_yr):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xlabel("Year", color=P["text"])
    ax3.set_ylabel("W/kg", color=P["text"])
    ax3.set_title("W/kg Distribution by Year  |  Red = surgery year", fontweight="bold")
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    out = Path(save_dir) / "wkg_progression.png"
    plt.savefig(out, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  ✅ W/kg progression saved → {out}")


if __name__ == "__main__":
    from src.data_loader import load
    df = load()
    print("\n" + "="*55)
    print("  W/KG PROGRESSION ANALYSIS")
    print("="*55)
    plot_wkg_progression(df)
