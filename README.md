# 🚴 Cycling Performance ML

> **6+ years · 1030 sessions · 2,447 hours · real athlete data**
> Road cycling with power meter · Catalonia, Spain · Post-surgery rebuild (Jun 2025)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://cycling-performance-ml-gwl7kzbkctmdgnatg2jvnt.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Dashboard

**[Open Dashboard →](https://cycling-performance-ml-gwl7kzbkctmdgnatg2jvnt.streamlit.app/)**

Real-time cycling performance dashboard built with Streamlit. Updates automatically via Intervals.icu API pipeline.

---

## 📁 Project Structure

```
cycling-performance-ml/
├── data/                          # Data files (gitignored)
│   ├── JOIN_STRAVA_TP.xlsx        # Historical joined dataset
│   ├── combined_training_data.csv # Merged API + historical
│   └── wellness_data.csv          # Intervals.icu wellness
├── src/
│   ├── config.py                  # Constants, zones, colours, athlete profile
│   ├── data_loader.py             # Load + clean + feature engineering
│   ├── pmc.py                     # Performance Management Chart (CTL/ATL/TSB)
│   ├── ftp_analysis.py            # Random Forest — what drives FTP gains
│   ├── wkg_progression.py         # W/kg regression + 12-week forecast
│   ├── clustering.py              # K-Means + PCA — session archetypes
│   ├── fatigue_detection.py       # Isolation Forest — overreach detection
│   └── intervals_api.py           # Live Intervals.icu API pipeline
├── app.py                         # Streamlit dashboard
├── main.py                        # Run full ML pipeline
├── METRICS_GLOSSARY.md            # Every metric explained in plain language
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Features

### Today's Status
- CTL / ATL / TSB with 7-day deltas
- eFTP from Intervals.icu
- W/kg 10-session rolling average
- Colour-coded fatigue state with automatic recommendation

### 🔬 Objective Fatigue Signals
*Why objective metrics beat RPE for this athlete: heat, sleep, and motivation contaminate perceived effort. W/BPM doesn't lie.*

- **28-day rolling W/BPM baseline** — your personal efficiency benchmark
- **Session dots coloured vs baseline** — green = adapting, red = fatigued
- **Hot session detection** (>28°C) — orange marker so heat isn't misread as fatigue
- **Automatic fatigue alert** — if W/BPM drops >5% below baseline, dashboard flags it regardless of TSB
- **Fatigue overrides recommendation** — physiological signal beats TSB number

### 🇳🇴 Norwegian Method Compliance
- Weekly true threshold sessions (IF ≥ 0.85) vs Z3 drift (IF 0.75–0.85)
- IF distribution histogram — validates training polarisation
- Target: 2 threshold sessions/week + pure Z2 everything else

### 📈 Performance Management Chart
- CTL / ATL with correct EWM formula (42-day / 7-day)
- TSB coloured by fatigue state
- Weekly TSS bars with overreach reference lines

### 🎯 FTP Development Analysis
- FTP Stimulus Score by training type (IF² × Duration × 100)
- TSS split: quality vs base volume
- Weekly FTP stimulus trend with 4-week rolling average

### ⚡ Power Curve
- Best efforts at 5s / 1min / 5min / 10min / 20min / 30min / 60min
- W/kg at each duration with target reference lines
- Identifies the gap between neuromuscular ceiling and aerobic threshold

### 📈 FTP Progression
- Monthly estimated FTP (best NP × 0.95) from 2019 to present
- Surgery structural break annotated
- Annual peak FTP comparison

### 🏔️ Elevation Stats
- Weekly elevation with 3000m/week target line
- Annual elevation totals
- Career total and best single session

### 📊 Year vs Year Comparison
- This year vs same period last year: TSS, sessions, power, W/kg, elevation, hours
- Monthly TSS comparison chart
- Monthly W/kg comparison chart

---

## 🔬 ML Analyses

| Module | Method | Question answered |
|---|---|---|
| `ftp_analysis.py` | Random Forest + feature importance | Which training types most drive FTP? |
| `wkg_progression.py` | Linear regression + Ridge | Where is W/kg heading? |
| `clustering.py` | K-Means + PCA | Hidden session archetypes? |
| `fatigue_detection.py` | Isolation Forest | When is fatigue genuine overreach? |

---

## 🚀 Quick Start

```bash
git clone https://github.com/rubenfm77/cycling-performance-ml.git
cd cycling-performance-ml
pip install -r requirements.txt

# Add your data
cp /path/to/JOIN_STRAVA_TP.xlsx data/

# Set up Intervals.icu API credentials
echo "INTERVALS_ATHLETE_ID=your_id" > .env
echo "INTERVALS_API_KEY=API_KEY:your_key" >> .env

# Fetch latest data
python src/intervals_api.py

# Run dashboard
streamlit run app.py

# Run ML pipeline
python main.py
```

---

## 📋 Weekly Workflow

Every Sunday after your long ride:
```bash
python src/intervals_api.py   # fetch fresh data
streamlit run app.py          # open dashboard
```

Check:
1. TSB — am I fresh or fatigued?
2. W/BPM vs baseline — objective fatigue signal
3. Norwegian compliance — 2 quality sessions done?
4. FTP stimulus — is training mix driving FTP up?

---

## 🏆 Key ML Findings

- **FTP stimulus** best predicted by `IF² × duration` — not intensity alone
- **Duration beats intensity**: RF assigns 49% importance to TSS, only 1.4% to IF alone
- **PIRAMIDAL + FTP + SST** cluster together as highest-adaptation archetype
- **Isolation Forest** correctly flags overreach periods and post-surgery detraining
- **W/BPM** is the most reliable fatigue signal — outperforms TSB alone
- **Z3 drift** (IF 0.75–0.85) is the biggest training mistake for amateur cyclists — high fatigue, low adaptation

---

## ⚙️ Data Sources

- **Strava** → distance, elevation, speed, HR
- **TrainingPeaks** → TSS, IF, NP, power zones
- **Intervals.icu API** → eFTP, CTL, ATL, efficiency, wellness
- **Wahoo ELEMNT** → raw power, L/R balance, temperature, cadence

---

## 📖 Metrics Reference

See **[METRICS_GLOSSARY.md](METRICS_GLOSSARY.md)** for a plain-language explanation of every metric in the dashboard — what it is, why it matters, and how to monitor it.

---

## ⚠️ Data Privacy

Source data files are excluded via `.gitignore`. Only code and pre-generated charts are public.

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
streamlit>=1.28
plotly>=5.17
requests>=2.31
python-dotenv>=1.0
```

---

*Catalonia, Spain · 2019–2026 · rubenfm77*
