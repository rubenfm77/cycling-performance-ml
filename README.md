# Cycling Performance ML

> **6+ years · 1030 sessions · 2,447 hours · real athlete data**
> Road cycling with power meter · Catalonia, Spain · Post-surgery rebuild (Jun 2025)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://cycling-performance-ml-gwl7kzbkctmdgnatg2jvnt.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Dashboard

**[Open Dashboard →](https://cycling-performance-ml-gwl7kzbkctmdgnatg2jvnt.streamlit.app/)**

Real-time cycling performance dashboard built with Streamlit. Updates automatically via Intervals.icu API pipeline.

---

## Project Structure

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
└── requirements.txt
```

---

## Dashboard Features

### Today's Status
- CTL / ATL / TSB with 7-day deltas
- eFTP from Intervals.icu
- W/kg 10-session rolling average
- Colour-coded fatigue state with automatic recommendation

### Objective Fatigue Signals

*Why objective metrics beat RPE for this athlete: heat, sleep, and motivation contaminate perceived effort. W/BPM doesn't lie.*

- **28-day rolling W/BPM baseline** — personal efficiency benchmark
- **Session dots coloured vs baseline** — green = adapting, red = fatigued
- **Hot session detection** (>28°C) — orange marker so heat isn't misread as fatigue
- **Automatic fatigue alert** — if W/BPM drops >5% below baseline, dashboard flags it regardless of TSB
- **Fatigue overrides recommendation** — physiological signal beats TSB number

### Norwegian Method Compliance
- Weekly true threshold sessions (IF ≥ 0.85) vs Z3 drift (IF 0.75–0.85)
- IF distribution histogram — validates training polarisation
- Target: 2 threshold sessions/week + pure Z2 everything else

### Performance Management Chart
![PMC](pmc_full.png)

- CTL / ATL with correct EWM formula (42-day / 7-day)
- TSB coloured by fatigue state
- Weekly TSS bars with overreach reference lines

### FTP Development Analysis
![FTP](ftp_analysis.png)

- FTP Stimulus Score by training type (`IF² × Duration × 100`)
- TSS split: quality vs base volume
- Weekly FTP stimulus trend with 4-week rolling average

### Session Clustering
![Clustering](clustering.png)

- K-Means + PCA reveals hidden session archetypes from 1030 rides
- Maps which training types actually cluster with performance gains

### Fatigue Detection
![Fatigue](fatigue_detection.png)

- Isolation Forest flags overreach and post-surgery detraining episodes
- Cross-validated against the W/BPM objective signal

---

## ML Analyses

| Module | Method | Question answered |
|---|---|---|
| `ftp_analysis.py` | Random Forest + feature importance | Which training types most drive FTP? |
| `wkg_progression.py` | Linear regression + Ridge | Where is W/kg heading? |
| `clustering.py` | K-Means + PCA | Hidden session archetypes? |
| `fatigue_detection.py` | Isolation Forest | When is fatigue genuine overreach? |

---

## Quick Start

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

## Data Sources

| Source | Data provided |
|---|---|
| **Strava** | Distance, elevation, speed, HR |
| **TrainingPeaks** | TSS, IF, NP, power zones |
| **Intervals.icu API** | eFTP, CTL, ATL, efficiency, wellness |
| **Wahoo ELEMNT** | Raw power, L/R balance, temperature, cadence |

---

## Conclusions

Six years and four ML models converge on a clear picture of what actually drives cycling performance — and where most amateur athletes go wrong.

**FTP is built through volume, not isolated intensity**
The Random Forest assigns **49% importance to TSS** (training volume) and only 1.4% to IF (intensity) as a standalone predictor. The best predictor of FTP improvement is the **FTP Stimulus Score** (`IF² × duration × 100`) — sessions that combine quality with duration consistently precede breakthroughs. Duration beats intensity every time.

**The biggest training mistake at amateur level is Z3 drift**
Z3 drift (IF 0.75–0.85) is the most common training pattern in the dataset and the least productive. It accumulates fatigue faster than pure Z2 but provides far less adaptation stimulus than genuine threshold work (IF ≥ 0.85). Most "easy" rides that drift into Z3 would produce better results ridden 10–15 W lower and kept in Z2.

**The PIRAMIDAL + FTP + SST cluster is the highest-adaptation archetype**
K-Means identifies four distinct session types. Sessions that combine polarised intensity (threshold blocks + Z2 base) cluster with the largest subsequent FTP gains. Pure Z2 sessions cluster with recovery and base-building. Z3 drift sessions cluster with fatigue accumulation without proportional adaptation.

**W/BPM outperforms TSB as a fatigue signal**
TSB (Training Stress Balance) can show a positive "fresh" number on days when cardiac efficiency has not recovered. The W/BPM efficiency ratio (watts per heartbeat) is more reliable: when it drops more than 5% below the 28-day rolling baseline, performance is degraded regardless of what TSB says. This matters most in the week after hard training blocks.

**Post-surgery recovery follows a predictable arc**
The Isolation Forest correctly identifies all major overreach episodes and the surgery-induced detraining arc. The Ridge regression model projects W/kg recovery to pre-surgery levels within 6 months — conditional on maintaining Norwegian-compliant training load (2 quality sessions per week, Z2 everything else).

**Practical upshot**
Measure your W/BPM. Eliminate Z3. Train threshold twice a week at IF ≥ 0.85. Keep everything else genuinely easy. The data confirms what the literature says but makes it personal — with numbers from 1030 rides over six years.

---

## Metrics Reference

See **[METRICS_GLOSSARY.md](METRICS_GLOSSARY.md)** for a plain-language explanation of every metric — what it is, why it matters, and how to monitor it.

---

## Data Privacy

Source data files are excluded via `.gitignore`. Only code and pre-generated charts are public.

---

*Catalonia, Spain · 2019–2026 · rubenfm77*
