# src/intervals_api.py
"""
Intervals.icu API pipeline.
Fetches activities, wellness, and eFTP data automatically.
Merges with historical Excel data into a unified dataset.
"""

import os
import re
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
        "description":            "icu_description",
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

        api_df = api_df.copy()

        # ── Label source 1: the Intervals.icu Description field ───────────────
        # You type the training type (e.g. "END", "FTP", "VO2MAX") into the
        # activity Description in Intervals.icu. The API returns it as
        # icu_description. We match any known label as a standalone word.
        KNOWN_TYPES = [
            "AEROBIC BASE", "VO2MAX", "VO2 MAX", "Q-I INTERVALS", "FATMAX",
            "PIRAMIDAL", "BILLAT", "TORQUE", "TEMPO", "END", "FTP", "SST",
        ]

        def _label_from_text(text) -> "str | float":
            if not isinstance(text, str) or not text.strip():
                return np.nan
            hay = " " + re.sub(r"[^A-Z0-9 ]", " ", text.upper())
            hay = re.sub(r"\s+", " ", hay) + " "
            for t in sorted(KNOWN_TYPES, key=len, reverse=True):
                tok = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", t)).strip()
                if f" {tok} " in hay:
                    return "VO2MAX" if t == "VO2 MAX" else t
            return np.nan

        if "icu_description" in api_df.columns:
            api_df["training_type"] = api_df["icu_description"].apply(_label_from_text)
        else:
            api_df["training_type"] = np.nan

        # ── Label source 2: fall back to the historical Excel label by date ───
        if "training_type" in hist.columns:
            label_by_date = (
                hist.dropna(subset=["training_type"])
                    .assign(_d=hist["date"].astype(str))
                    .drop_duplicates("_d", keep="last")
                    .set_index("_d")["training_type"]
            )
            api_df["_d"] = api_df["date"].astype(str)
            api_df["training_type"] = api_df["training_type"].fillna(
                api_df["_d"].map(label_by_date)
            )
            api_df = api_df.drop(columns=["_d"])

        n_desc = api_df["training_type"].notna().sum()
        print(f"  🏷️  Labelled {n_desc}/{len(api_df)} API rows "
              f"(Description field + historical fallback)")
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


def fetch_power_curve(days_back: int = 365) -> pd.DataFrame:
    """
    Fetch the real mean-maximal power (MMP) curve from Intervals.icu.

    This is the *true* power curve — best power sustained for each duration
    across all rides — not the broken whole-ride-NP approximation. The endpoint
    is /athlete/{id}/power-curves. The response shape can vary slightly between
    Intervals.icu versions, so we parse defensively and fall back to an empty
    DataFrame (the app then hides the chart) rather than crash.
    """
    print("  Fetching power curve (MMP)...")
    # Standard durations we want to display, in seconds
    want = {
        5:    "5s",
        60:   "1 min",
        300:  "5 min",
        600:  "10 min",
        1200: "20 min",
        1800: "30 min",
        3600: "60 min",
    }
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to   = datetime.now().strftime("%Y-%m-%d")

    data = None
    for params in (
        {"curves": "all", "type": "Ride", "newest": date_to, "oldest": date_from},
        {"type": "Ride", "newest": date_to, "oldest": date_from},
        {"type": "Ride"},
    ):
        try:
            data = _get(f"athlete/{ATHLETE_ID}/power-curves", params=params)
            if data:
                break
        except Exception as e:
            print(f"    power-curves attempt failed ({params}): {e}")
            data = None

    if not data:
        print("  ⚠️  No power-curve data returned — chart will be hidden")
        return pd.DataFrame()

    # The response is typically a list of curve objects, each with parallel
    # arrays "secs" (durations) and "values"/"watts" (best power at each).
    # Pick the curve with the most coverage and extract our target durations.
    def _extract(curve):
        secs = curve.get("secs") or curve.get("seconds") or curve.get("x")
        vals = (curve.get("values") or curve.get("watts")
                or curve.get("y") or curve.get("power"))
        if not secs or not vals or len(secs) != len(vals):
            return None
        lookup = dict(zip(secs, vals))
        rows = []
        for sec, label in want.items():
            # nearest available duration within 10%
            best = None
            for s in secs:
                if abs(s - sec) <= max(2, sec * 0.10):
                    if best is None or abs(s - sec) < abs(best - sec):
                        best = s
            w = lookup.get(best) if best is not None else None
            if w is not None and w > 0:
                rows.append({"secs": sec, "duration": label,
                             "watts": float(w), "wkg": float(w) / WEIGHT_KG})
        return rows

    # The real Intervals.icu response wraps the curves under "list":
    #   {"list": [{"secs": [...], "values": [...], "watts": [...]}, ...]}
    # Older/other shapes may return a bare list or a single dict. Handle all.
    if isinstance(data, dict) and "list" in data and isinstance(data["list"], list):
        curves = data["list"]
    elif isinstance(data, list):
        curves = data
    else:
        curves = [data]

    best_rows = []
    for c in curves:
        if not isinstance(c, dict):
            continue
        r = _extract(c)
        if r and len(r) > len(best_rows):
            best_rows = r

    if not best_rows:
        print("  ⚠️  Power-curve response shape not recognised — chart hidden")
        return pd.DataFrame()

    pc = pd.DataFrame(best_rows).sort_values("secs").reset_index(drop=True)

    # Power meter over-reading correction (Oct 2024–Dec 2025).
    # The MMP endpoint returns aggregate bests without per-effort dates, so we
    # can't selectively correct only the affected efforts. Since the 365-day
    # fetch window overlaps heavily with the affected period and inflated efforts
    # would always beat clean ones in a best-of ranking, apply 0.855 to the
    # whole curve to bring it in line with the corrected session data.
    POWER_CORRECTION = 0.855
    pc["watts"] = (pc["watts"] * POWER_CORRECTION).round(1)
    pc["wkg"]   = (pc["watts"] / WEIGHT_KG).round(4)

    print(f"  ✅ Power curve: {len(pc)} durations "
          f"({pc['watts'].min():.0f}–{pc['watts'].max():.0f}W, correction={POWER_CORRECTION})")
    return pc


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
    power_curve = fetch_power_curve(days_back)

    excel_path = "data/JOIN_STRAVA_TP.xlsx"
    if Path(excel_path).exists():
        combined = merge_with_historical(activities, excel_path)
    else:
        combined = activities
        print("  ⚠️  No Excel file found — using API data only")

    save_combined(combined)

    if not power_curve.empty:
        power_curve.to_csv("data/power_curve.csv", index=False)
        print(f"  ✅ Power curve saved → data/power_curve.csv")

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
