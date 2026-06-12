"""
Bulk-upload Strava archive activity files to Intervals.icu.

Usage:
    1. Extract the Strava export ZIP (WinRAR: "Extract To").
    2. Fill in ATHLETE_ID and API_KEY below.
    3. Set ACTIVITIES_DIR to the extracted 'activities' folder.
    4. Run:  python upload_to_intervals.py

Duplicates already in Intervals.icu are skipped automatically.
Safe to stop (Ctrl+C) and re-run — it will just skip what's done.
"""

import time
from pathlib import Path

import requests

# ── EDIT THESE THREE LINES ────────────────────────────────────────────────────
ATHLETE_ID = "i469810"
API_KEY = "PASTE-YOUR-API-KEY-HERE"
ACTIVITIES_DIR = r"C:\Users\levod\Downloads\export_112598 (1)\activities"
# ──────────────────────────────────────────────────────────────────────────────

VALID_EXT = {".fit", ".gz", ".tcx", ".gpx"}
URL = f"https://intervals.icu/api/v1/athlete/{ATHLETE_ID}/activities"


def main():
    folder = Path(ACTIVITIES_DIR)
    if not folder.exists():
        print(f"ERROR: folder not found: {folder}")
        print("Edit ACTIVITIES_DIR at the top of this script.")
        return

    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXT
    )
    print(f"Found {len(files)} activity files in {folder}\n")
    if not files:
        return

    ok, dup, fail = 0, 0, 0
    for i, f in enumerate(files, 1):
        try:
            with open(f, "rb") as fh:
                r = requests.post(
                    URL,
                    files={"file": (f.name, fh)},
                    auth=("API_KEY", API_KEY),
                    timeout=60,
                )
            if r.status_code in (200, 201):
                ok += 1
                status = "uploaded"
            elif r.status_code in (409, 422) or "uplicate" in r.text:
                dup += 1
                status = "duplicate (skipped)"
            else:
                fail += 1
                status = f"FAILED ({r.status_code}): {r.text[:80]}"
        except Exception as e:
            fail += 1
            status = f"ERROR: {e}"

        print(f"[{i}/{len(files)}] {f.name}: {status}")
        time.sleep(0.5)  # be polite to the API

    print("\n──────────────────────────────")
    print(f"Uploaded : {ok}")
    print(f"Skipped  : {dup} (already in Intervals.icu)")
    print(f"Failed   : {fail}")
    if fail:
        print("Re-run the script to retry failures — done files are skipped.")
    print("Intervals.icu will now recompute power curves and eFTP "
          "over the full history. Give it a while, then hit Refresh Data "
          "in the dashboard.")


if __name__ == "__main__":
    main()
