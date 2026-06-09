# src/intervals_api.py
"""
Intervals.icu API pipeline.
Fetches activities, wellness, and eFTP data automatically.
Merges with historical Excel data into a unified dataset.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ATHLETE_ID = os.getenv("INTERVALS_ATHLETE_ID")
API_KEY    = os.getenv("INTERVALS_API_KEY")
BASE_URL   = "https://intervals.icu/api/v1"
WEIGHT_KG  = 57.0


def _get_key():
    if not API_KEY:
        return ""
    return API_KEY.replace("API_KEY:", "").strip()


def _get(endpoint: str, params: dict = None):
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, auth=("API_KEY", _get_key()), params=params)
    r.raise_for_status()
    return r.json()


def fetch_activities(days_back: int = 365) -> pd.DataFrame:
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00")
    date_to   = datetime.now().strftime("%Y-%m-%dT23:59:59")
    print(f"  Fetching activities (last {days_back} days)...")

    data = _get(
        f"athlete/{ATHLETE_ID}/activities",
        params={"oldest": date_from, "newest": date_to}
    )

    if not data:
        print("  No activities found")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    print(f"  Raw columns from API: {list(df.columns[:20])}...")

    rename_map = {
        "start_date_local":       "date",
        "name":                   "activity_name",
        "type":                   "activity_type",
        "moving_time":            "duration_secs",
        "distance":               "distance_m",
        "total_elevation_gain":   "elevation",
        "average_watts":          "power_avg",
        "weighted_average_watts": "power_np",
        "max_watts":              "power_max",
        "average_heartrate":      "hr_avg",
        "max_heartrate":          "hr_max",
        "average_cadence":        "cadence",
        "average_temp":           "temp_avg",
        "icu_training_load":      "tss",
        "icu_intensity":          "if_score",
        "icu_eftp":               "eftp",
        "icu_fitness":            "ctl",
        "icu_fatigue":            "atl",
        "icu_form":               "tsb",
        "icu_efficiency_factor":  "efficiency",
        "icu_average_watts":      "power_avg_icu",
        "icu_normalized_watts":   "power_np_icu",
        "icu_variability_index":  "variability",
        "icu_power_hr":           "power_hr_ratio",
    }

    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)
    print(f"  Renamed {len(existing_renames)} columns")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    elif "start_date" in df.columns:
        df["date"] = pd.to_datetime(df["start_date"]).dt.date

    if "duration_secs" in df.columns:
        df["duration_h"] = df["duration_secs"] / 3600

    if "power_avg" not in df.columns and "power_avg_icu" in df.columns:
        df["power_avg"] = df["power_avg_icu"]
    if "power_np" not in df.columns and "power_np_icu" in df.columns:
        df["power_np"] = df["power_np_icu"]

    if "power_avg" in df.columns:
        df["w_per_kg"] = pd.to_numeric(df["power_avg"], errors="coerce") / WEIGHT_KG
    else:
        df["w_per_kg"] = np.nan

    if "power_avg" in df.columns and "hr_avg" in df.columns:
        p = pd.to_numeric(df["power_avg"], errors="coerce")
        h = pd.to_numeric(df["hr_avg"], errors="coerce")
        df["efficiency"] = np.where(h > 0, p / h, np.nan)

    if "tsb" not in df.columns and "ctl" in df.columns and "atl" in df.columns:
        df["tsb"] = pd.to_numeric(df["ctl"], errors="coerce") - \
                    pd.to_numeric(df["atl"], errors="coerce")

    if "if_score" in df.columns and "duration_h" in df.columns:
        i = pd.to_numeric(df["if_score"], errors="coerce")
        d = pd.to_numeric(df["duration_h"], errors="coerce")
        df["ftp_stimulus"] = (i ** 2) * d * 100

    print(f"  ✅ Fetched {len(df)} activities")
    print(f"  Final columns: {[c for c in df.columns if c in ['date','tss','if_score','power_avg','hr_avg','ctl','atl','tsb','eftp','w_per_kg']]}")
    return df


def fetch_wellness(days_back: int = 365) -> pd.DataFrame:
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to   = datetime.now().strftime("%Y-%m-%d")
    print(f"  Fetching wellness data...")

    data = _get(
        f"athlete/{ATHLETE_ID}/wellness",
        params={"oldest": date_from, "newest": date_to}
    )

    if not data:
        return pd.DataFrame()

    df = pd.json_normalize(data)

    if "id" in df.columns:
        df["date"] = pd.to_datetime(df["id"]).dt.date

    rename_map = {
        "restingHR":    "resting_hr",
        "hrv":          "hrv",
        "hrvSDNN":      "hrv_sdnn",
        "weight":       "weight",
        "sleepSecs":    "sleep_secs",
        "sleepScore":   "sleep_score",
        "fatigue":      "wellness_fatigue",
        "mood":         "mood",
        "motivation":   "motivation",
        "kcalConsumed": "kcal",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)

    if "sleep_secs" in df.columns:
        df["sleep_h"] = pd.to_numeric(df["sleep_secs"], errors="coerce") / 3600

    print(f"  ✅ Fetched {len(df)} wellness days")
    return df


def merge_with_historical(
    api_df: pd.DataFrame,
    excel_path: str = "data/JOIN_STRAVA_TP.xlsx"
) -> pd.DataFrame:
    print("\n  Merging API data with historical Excel...")

    hist = pd.read_excel(excel_path)
    hist["date"] = pd.to_datetime(hist["Activity Date"]).dt.date

    hist_rename = {
        "TRAINING_TYPE":          "training_type",
        "TSS":                    "tss",
        "IF":                     "if_score",
        "PowerAverage":           "power_avg",
        "Weighted Average Power": "power_np",
        "TimeTotalInHours":       "duration_h",
        "HeartRateAverage":       "hr_avg",
        "WEIGHT_KG":              "weight",
        "Elevation Gain":         "elevation",
        "DistanceInMeters":       "distance_m",
        "Average Cadence":        "cadence",
        "Average Temperature":    "temp_avg",
    }
    hist = hist.rename(columns={k: v for k, v in hist_rename.items() if k in hist.columns})

    if not api_df.empty and "date" in api_df.columns:
        api_dates = set(api_df["date"].astype(str))
        hist_filtered = hist[~hist["date"].astype(str).isin(api_dates)]
    else:
        hist_filtered = hist

    combined = pd.concat([hist_filtered, api_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # ── Power meter correction — Oct 2024 to Dec 2025 ────────────────────────
    combined["date_str"] = combined["date"].astype(str)
    mask = (
        (combined["date_str"] >= "2024-10-09") &
        (combined["date_str"] < "2025-12-01") &
        (combined["power_avg"].notna())
    )
    combined["power_adj"] = combined["power_avg"].copy()
    combined.loc[mask, "power_adj"] = combined.loc[mask, "power_avg"] * 0.855
    combined = combined.drop(columns=["date_str"])
    print(f"  ⚡ Power correction applied to {mask.sum()} sessions")

    print(f"  Historical : {len(hist_filtered)} sessions")
    print(f"  API (new)  : {len(api_df)} sessions")
    print(f"  Combined   : {len(combined)} sessions")

    return combined


def save_combined(df: pd.DataFrame,
                  path: str = "data/combined_training_data.csv") -> None:
    df.to_csv(path, index=False)
    print(f"  ✅ Saved → {path}")


def run_pipeline(days_back: int = 365) -> pd.DataFrame:
    print("\n" + "="*55)
    print("  INTERVALS.ICU API PIPELINE")
    print("="*55)

    if not ATHLETE_ID or not API_KEY:
        raise ValueError(
            "Missing credentials. Check your .env file:\n"
            "  INTERVALS_ATHLETE_ID=i469810\n"
            "  INTERVALS_API_KEY=API_KEY:your_key_here"
        )

    activities = fetch_activities(days_back)
    wellness   = fetch_wellness(days_back)

    excel_path = "data/JOIN_STRAVA_TP.xlsx"
    if Path(excel_path).exists():
        combined = merge_with_historical(activities, excel_path)
    else:
        combined = activities
        print("  ⚠️  No Excel file found — using API data only")

    save_combined(combined)

    if not wellness.empty:
        wellness.to_csv("data/wellness_data.csv", index=False)
        print(f"  ✅ Wellness saved → data/wellness_data.csv")

        if "hrv" in wellness.columns:
            hrv = wellness["hrv"].dropna()
            if len(hrv) > 0:
                print(f"\n  HRV ({len(hrv)} days): mean={hrv.mean():.1f} "
                      f"min={hrv.min():.1f} max={hrv.max():.1f}")

    print(f"\n  ✅ Pipeline complete — run python main.py for ML analysis\n")
    return combined


if __name__ == "__main__":
    df = run_pipeline(days_back=365)
    key_cols = [c for c in ["date", "tss", "if_score", "power_avg",
                             "power_adj", "hr_avg", "ctl", "atl", "tsb",
                             "eftp", "w_per_kg", "training_type"]
                if c in df.columns]
    print(df[key_cols].tail(10).to_string())
