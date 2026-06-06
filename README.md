# 🚴 Cycling Performance ML

> **Personal cycling analytics project** — 6+ years of training data (2019–2026), 944 sessions.  
> Power · FTP · Fatigue · Clustering · W/kg progression · Anomaly detection.

This is a **real athlete dataset** — not synthetic. 57kg climber, post-surgery rebuild (Jun 2025), training on road with power meter (Wahoo + TrainingPeaks + Intervals.icu stack).

---

## 📁 Project Structure

```
cycling-performance-ml/
│
├── data/
│   └── JOIN_STRAVA_TP.xlsx          # Source data (Strava + TP + ICU join)
│
├── src/
│   ├── config.py                    # Constants, FTP, zones, colours
│   ├── data_loader.py               # Load + clean + engineer features
│   ├── pmc.py                       # Performance Management Chart (CTL/ATL/TSB)
│   ├── ftp_analysis.py              # Best training types for FTP (ML)
│   ├── wkg_progression.py           # W/kg time-series + regression forecast
│   ├── clustering.py                # Session clustering (K-Means + PCA)
│   ├── fatigue_detection.py         # Anomaly detection (Isolation Forest)
│   └── visualisation.py             # Shared plotting utilities
│
├── outputs/                         # Generated charts (auto-created)
│
├── main.py                          # Run full pipeline
├── requirements.txt
└── README.md
```

---

## 🔬 Analyses

| Module | Method | Question answered |
|---|---|---|
| `ftp_analysis.py` | Random Forest + feature importance | Which training types most drive FTP gains? |
| `wkg_progression.py` | Linear regression + Ridge + time-series | Where is W/kg heading? |
| `clustering.py` | K-Means + PCA + silhouette | Are there hidden session archetypes? |
| `fatigue_detection.py` | Isolation Forest + EWM PMC | When is fatigue genuine overreach vs normal load? |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/cycling-performance-ml.git
cd cycling-performance-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data
cp /path/to/JOIN_STRAVA_TP.xlsx data/

# 4. Run full pipeline
python main.py

# 5. View outputs
ls outputs/
```

---

## 📊 Key Findings

- **FTP stimulus** is best predicted by `IF² × duration` — not intensity alone
- **PIRAMIDAL + FTP + SST** sessions cluster together as the highest-adaptation archetype
- **Isolation Forest** correctly flags 2 overreach periods matching known fatigue events
- **W/kg trend** shows clear 2024 peak (3.30 W/kg avg) with 2025 dip post-surgery
- **6 session clusters** found: Recovery · Endurance · Tempo · Threshold · VO2 · Long Ride

---

## ⚙️ Data Sources

- **Strava** export (GPX metadata, HR, cadence, speed)
- **TrainingPeaks** sync (TSS, IF, NP, power zones, planned workouts)
- **Intervals.icu** API (eFTP, CTL, ATL, form, efficiency)
- **Wahoo ELEMNT** head unit (raw power, L/R balance, temperature)

---

## 📋 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
openpyxl>=3.1
```

---

## ⚠️ Notes

- ICU-enriched data (eFTP, CTL/ATL from API) only available for last 20 sessions (Feb–Apr 2026)
- FTP proxy used for full dataset: `IF² × duration × 100` (validated against known FTP tests)
- Surgery June 2025 creates a structural break in the time series — handled explicitly in models
- Weight range: 55–57kg across career

---

*Data: personal · Analysis: open source · PRs welcome*
