# src/data_loader.py
"""
Load, clean, and engineer features from JOIN_STRAVA_TP.xlsx.
All downstream modules import from here — single source of truth.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import ATHLETE, CTL_SPAN, ATL_SPAN, MAIN_TYPES


def _ewm(series: pd.Series, span: int) -> pd.Series:
    """Exponential weighted mean — the correct PMC formula."""
    return series.ewm(span=span, adjust=False).mean()


def load_raw(path: str = "data/JOIN_STRAVA_TP.xlsx") -> pd.DataFrame:
    """Load the raw Excel join file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}.\n"
            "Copy JOIN_STRAVA_TP.xlsx into the data/ directory."
        )
    df = pd.read_excel(p)
    print(f"✅ Loaded {len(df)} rows · {df.shape[1]} columns")
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline. Returns analysis-ready DataFrame."""

    out = df.copy()

    # ── Dates ─────────────────────────────────────────────────────────────────
    out["date"]       = pd.to_datetime(out["Activity Date"])
    out               = out.sort_values("date").reset_index(drop=True)
    out["year"]       = out["date"].dt.year
    out["month"]      = out["date"].dt.to_period("M")
    out["week"]       = out["date"].dt.to_period("W")
    out["day_of_week"]= out["date"].dt.day_name()
    out["month_num"]  = out["date"].dt.month
    out["days_since_start"] = (out["date"] - out["date"].min()).dt.days

    # ── Surgery flag ──────────────────────────────────────────────────────────
    surgery = pd.Timestamp(ATHLETE["surgery_date"])
    out["post_surgery"] = (out["date"] > surgery).astype(int)
    out["days_since_surgery"] = (out["date"] - surgery).dt.days.clip(lower=0)

    # ── Power ─────────────────────────────────────────────────────────────────
    out["power_avg"]  = out["PowerAverage"].astype(float)
    out["power_np"]   = out["Weighted Average Power"].astype(float)
    out["power_max"]  = out["PowerMax"].astype(float)
    out["w_per_kg"]   = out["power_avg"] / out["WEIGHT_KG"]
    out["np_per_kg"]  = out["power_np"] / out["WEIGHT_KG"]

    # ── Load & intensity ──────────────────────────────────────────────────────
    out["tss"]        = out["TSS"].astype(float)
    out["if_score"]   = out["IF"].astype(float)
    out["duration_h"] = out["TimeTotalInHours"].astype(float)
    out["ftp_stimulus"] = (out["if_score"] ** 2) * out["duration_h"] * 100

    # ── Physiological ─────────────────────────────────────────────────────────
    out["hr_avg"]     = out["HeartRateAverage"].astype(float)
    out["hr_max"]     = out["HeartRateMax"].astype(float)
    out["weight"]     = out["WEIGHT_KG"].astype(float)
    out["efficiency"] = np.where(
        out["hr_avg"] > 0,
        out["power_avg"] / out["hr_avg"],
        np.nan
    )

    # ── Geography & environment ───────────────────────────────────────────────
    out["elevation"]  = out["Elevation Gain"].astype(float)
    out["distance_km"]= out["DistanceInMeters"].astype(float) / 1000
    out["elev_per_km"]= np.where(
        out["distance_km"] > 0,
        out["elevation"] / out["distance_km"],
        np.nan
    )
    out["temp_avg"]   = out["Average Temperature"].astype(float)
    out["cadence"]    = out["Average Cadence"].astype(float)

    # ── Power zones (minutes) ─────────────────────────────────────────────────
    for i in range(1, 7):
        out[f"z{i}_min"] = out[f"PWRZone{i}Minutes"].astype(float)

    # Zone totals and percentages
    zone_total = sum(out[f"z{i}_min"] for i in range(1, 7))
    for i in range(1, 7):
        out[f"z{i}_pct"] = np.where(zone_total > 0,
                                     out[f"z{i}_min"] / zone_total * 100, 0)

    out["z4plus_pct"] = out["z4_pct"] + out["z5_pct"] + out.get("z6_pct", 0)
    out["high_intensity_pct"] = out["z4plus_pct"]

    # ── Training type ─────────────────────────────────────────────────────────
    out["training_type"] = out["TRAINING_TYPE"].astype(str).str.strip()
    out["is_main_type"]  = out["training_type"].isin(MAIN_TYPES).astype(int)
    out["is_ftp_driver"] = out["training_type"].isin(
        ["FTP", "SST", "TEMPO", "PIRAMIDAL", "VO2MAX", "BILLAT", "Q-I INTERVALS"]
    ).astype(int)

    # ── PMC: CTL / ATL / TSB ──────────────────────────────────────────────────
    out["ctl"] = _ewm(out["tss"], CTL_SPAN)
    out["atl"] = _ewm(out["tss"], ATL_SPAN)
    out["tsb"] = out["ctl"] - out["atl"]

    # TSB state label
    conditions = [
        out["ctl"] < 30,
        out["tsb"] < -30,
        out["tsb"] < -10,
        out["tsb"] <   0,
        out["tsb"] <  10,
        out["tsb"] <  25,
    ]
    choices = [
        "Undertrained", "Overreached", "Deep Block",
        "Build Phase",  "Neutral",     "Fresh",
    ]
    out["fatigue_state"] = np.select(conditions, choices, default="Peak/Detrain Risk")

    # ── Rolling metrics ───────────────────────────────────────────────────────
    out["tss_7d"]     = out["tss"].rolling(7,  min_periods=1).sum()
    out["tss_28d"]    = out["tss"].rolling(28, min_periods=1).sum()
    out["power_7d"]   = out["power_avg"].rolling(7, min_periods=1).mean()
    out["eff_28d"]    = out["efficiency"].rolling(28, min_periods=1).mean()

    # ── Ramp rate ─────────────────────────────────────────────────────────────
    out["ramp_rate"]  = np.where(out["ctl"] > 0, out["atl"] / out["ctl"], np.nan)

    # ── W/kg rolling trend ────────────────────────────────────────────────────
    out["wkg_28d_avg"] = out["w_per_kg"].rolling(28, min_periods=5).mean()
    out["wkg_trend"]   = out["wkg_28d_avg"].diff(7)  # 7-session delta

    print(f"✅ Feature engineering complete · {out.shape[1]} features")
    return out


def get_main_types(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to main 11 training types only."""
    return df[df["training_type"].isin(MAIN_TYPES)].copy()


def get_quality_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to sessions with HR data and full power zones."""
    return df[
        df["hr_avg"].notna() &
        (df["power_avg"] > 0) &
        (df["duration_h"] > 0.5)
    ].copy()


def load(path: str = "data/JOIN_STRAVA_TP.xlsx") -> pd.DataFrame:
    """Convenience: load + clean in one call."""
    return clean_and_engineer(load_raw(path))


if __name__ == "__main__":
    df = load()
    print(df[["date", "training_type", "tss", "if_score",
              "power_avg", "w_per_kg", "ctl", "tsb",
              "fatigue_state"]].tail(10).to_string())
