# 📖 Cycling Performance Metrics Glossary
### Every metric in the dashboard — what it is, why it matters, how to use it

---

## 🏋️ TRAINING LOAD METRICS

### TSS — Training Stress Score
**What it is:** A number that represents how hard a session was, combining intensity and duration.
A 1-hour ride at exactly your FTP = 100 TSS. That's the baseline.

**Formula:** `(seconds × NP × IF) / (FTP × 3600) × 100`

**How to read it:**
| TSS | Session type |
|---|---|
| < 50 | Easy — recovery ride |
| 50–100 | Moderate — standard training session |
| 100–150 | Hard — quality session |
| 150–200 | Very hard — long ride with intensity |
| > 200 | Epic — full day in the mountains |

**Weekly totals:**
| Weekly TSS | Level |
|---|---|
| < 250 | Low — maintenance only |
| 250–400 | Moderate — building fitness |
| 400–600 | High — serious training |
| 600–800 | Very high — monitor fatigue |
| > 800 | Dangerous — overtraining risk |

**Limitation:** TSS treats all effort equally. A 3h Z2 ride and a 3h hard race can have the same TSS but feel completely different. Don't use TSS alone.

---

### IF — Intensity Factor
**What it is:** How hard the session was relative to your FTP. Expressed as a decimal.
`IF = Normalised Power / FTP`

**How to read it:**
| IF | Zone | Session type |
|---|---|---|
| < 0.55 | Z1 | Active recovery |
| 0.55–0.75 | Z2 | Endurance / aerobic base |
| 0.75–0.85 | Z3 | Tempo / SST |
| 0.85–0.95 | Z4 | Threshold — the money zone |
| 0.95–1.05 | Z5 | VO2max |
| > 1.05 | Z6–Z7 | Anaerobic / sprint |

**Why it matters for you:** The Norwegian Method says your quality sessions should land at IF 0.85–0.95. If they're at 0.78 you're doing Tempo (Z3), not threshold — less FTP stimulus. If they're at 0.97 you're doing VO2max — more fatigue, less specific FTP benefit.

---

### NP — Normalised Power
**What it is:** The power output you would have averaged if you'd ridden at perfectly constant effort, instead of the variable terrain of Castellar and Sant Llorenç.

**Why it matters:** Your average power on a hilly route (e.g. 170W avg) massively underestimates the actual physiological stress because of the constant accelerations and climbs. NP corrects for this. A session with avg 170W but NP 210W was actually as hard as riding at 210W constantly.

**Your typical NP vs Avg gap:** 15–30W. The rougher the terrain, the bigger the gap.

---

### CTL — Chronic Training Load (Fitness)
**What it is:** Your fitness level. A 42-day exponential weighted average of your daily TSS. Think of it as "how much training your body has absorbed over the last 6 weeks."

**How to read it:**
| CTL | Fitness level |
|---|---|
| < 30 | Beginner / detrained |
| 30–50 | Recreational |
| 50–70 | Trained amateur |
| 70–90 | Serious amateur |
| 90–120 | High-level amateur |
| > 120 | Elite / pro |

**Your current CTL: ~147** — this is high. You've built serious fitness even through the surgery rebuild. This is the legacy of your pre-accident years showing through.

**How to increase it:** Consistent training volume. CTL rises ~1 point per week with sustained load. Falls ~1 point per week of rest. You can't rush it — this is why 1 week of hard training doesn't change anything.

---

### ATL — Acute Training Load (Fatigue)
**What it is:** Your current fatigue. A 7-day exponential weighted average of TSS. Rises quickly when you train hard, falls quickly when you rest.

**How to read it:** ATL > CTL = you're in a fatigue hole. ATL < CTL = you're fresher than your fitness level. ATL spikes are normal after hard weeks — what matters is managing the difference.

**Your current ATL: ~165** — higher than your CTL, meaning you're currently carrying fatigue. Expected and correct in a build block.

---

### TSB — Training Stress Balance (Form)
**What it is:** `TSB = CTL - ATL`. The single number that tells you how fresh or fatigued you are right now.

**How to read it:**
| TSB | State | What it means |
|---|---|---|
| < -30 | 🔴 Overreached | Too much load — mandatory rest |
| -30 to -10 | 🟠 Deep Block | Hard training block — high adaptation |
| -10 to 0 | 🟡 Build Phase | Productive training zone |
| 0 to +10 | ⚪ Neutral | Balanced |
| +10 to +25 | 🟢 Fresh | Race-ready, peak quality sessions |
| > +25 | 🟣 Peak / Risk | Peak form — race or detraining begins |

**Critical insight:** TSB is lagged. If you trained hard Monday-Wednesday and check TSB Wednesday night, it hasn't fully updated yet. Always look at the trend, not just today's number.

**Your target:** Stay between -20 and -5 during build blocks. Peak to +10 to +20 before important events or FTP tests.

---

## ⚡ POWER METRICS

### FTP — Functional Threshold Power
**What it is:** The maximum power you can sustain for approximately 60 minutes. The most important number in cycling performance.

**How to test it:**
- 20-minute all-out effort × 0.95 = FTP
- Or use your dashboard's estimated FTP from NP data

**Your numbers:**
- Current FTP: ~235W (4.1 W/kg at 57kg)
- Pre-accident FTP: 275W (4.82 W/kg)
- Target: Return to 275W by mid-2027

**Why it matters:** Everything else is calculated relative to FTP. Your zones, your TSS, your IF — all depend on having the right FTP. If your FTP is set wrong in Intervals.icu, every other metric is wrong.

---

### eFTP — Estimated FTP
**What it is:** Intervals.icu's algorithm estimates your FTP from your recent ride data without needing a formal test. It looks at your best power outputs at various durations and calculates what your theoretical 60-minute power would be.

**How accurate is it:** Pretty good when you have lots of varied data. Underestimates after periods of low training (like your surgery recovery). Always treat eFTP as a floor, not a ceiling.

---

### W/kg — Watts per Kilogram
**What it is:** Your power output divided by your body weight. The most important metric for a climber.

`W/kg = Power (W) / Weight (kg)`

**Why weight matters so much for you:** On a climb, gravity is the main resistance. A 57kg rider at 235W produces 4.12 W/kg. A 75kg rider would need 309W to match that on a climb. This is why climbers are light.

**Reference values for climbers:**
| W/kg (FTP) | Level |
|---|---|
| < 2.5 | Beginner |
| 2.5–3.5 | Recreational |
| 3.5–4.0 | Trained amateur |
| 4.0–4.5 | Serious amateur / Cat 3-4 |
| 4.5–5.0 | High-level amateur / Cat 1-2 |
| 5.0–5.5 | Elite amateur / domestic pro |
| > 5.5 | Pro peloton |

**Your journey:** Pre-accident 4.82 W/kg → surgery → rebuilding toward 4.12 W/kg current → target 4.82+ W/kg

---

### Power Curve
**What it is:** Your best power output at every duration — from 5 seconds to 60 minutes. Shows your complete physiological profile.

**How to read it:** A true climber has a curve that drops more slowly at longer durations. A sprinter drops fast after 30 seconds but has a massive peak. Your curve should show strong 20-60 minute power relative to your sprint.

**The gap in your curve:** Your 3-minute power (277W) vs 20-minute power (224W) is a 53W drop in 17 minutes. That's too steep — it confirms your aerobic threshold is the limiter, not your neuromuscular capacity. FTP intervals directly target this gap.

---

## 🫀 PHYSIOLOGICAL METRICS

### Cardiac Efficiency (W/BPM)
**What it is:** How many watts your heart produces per beat. `Efficiency = Avg Power / Avg HR`

**Why it's the best objective fatigue metric for you:**
- Rises as you get fitter (same HR, more power)
- Drops when fatigued (same power, heart has to work harder)
- Drops in heat (HR rises due to thermoregulation, power stays same)
- Drops with poor sleep (HR higher at rest and during exercise)
- Completely objective — your perception doesn't contaminate it

**How to monitor it:**
- Calculate your personal baseline over 28 days
- If current session W/BPM is >5% below baseline → fatigue signal regardless of how you feel
- If current session W/BPM is >5% above baseline → adaptation happening, good week to push

**Your typical range:** 1.2–1.45 W/BPM depending on terrain and conditions

---

### HRRc — Heart Rate Recovery
**What it is:** How fast your heart rate drops in the first minute after stopping exercise. Measured in bpm.

**How to read it:**
| HRRc | Fitness level |
|---|---|
| < 12 bpm | Poor recovery — overtrained or ill |
| 12–20 bpm | Moderate |
| 20–30 bpm | Good |
| > 30 bpm | Excellent — well-trained |

**Your values from screenshots:** HRRc 36-48 bpm — excellent cardiovascular fitness.

---

### TRIMP — Training Impulse
**What it is:** An older metric similar to TSS but based purely on HR data rather than power. Used by Intervals.icu as a backup when power data is unavailable.

**For you:** Ignore it. You have a power meter. TSS is more accurate.

---

## 🧠 TRAINING METHODOLOGY METRICS

### FTP Stimulus Score
**What it is:** A composite metric I created for your dashboard.
`FTP Stimulus = IF² × Duration × 100`

**Why IF is squared:** Intensity has a non-linear effect on threshold adaptation. Going from IF 0.80 to 0.90 doesn't just add 12% more stimulus — it roughly doubles it. This is why 1 hour at threshold is worth more than 2 hours at Tempo.

**Ranking (your data):**
| Rank | Type | Score |
|---|---|---|
| 1 | FTP intervals | 160 |
| 2 | PIRAMIDAL | 153 |
| 3 | VO2MAX | 152 |
| 4 | SST | 149 |
| 5 | TEMPO | 147 |

**How to use it:** If your FTP Stimulus chart shows mostly grey bars (END/FATMAX/AEROBIC BASE), your training won't move your FTP. You need green bars (FTP/SST/TEMPO) to account for at least 30-40% of total TSS.

---

### Norwegian Method Compliance
**What it is:** A training philosophy from Norwegian endurance sport (Ingebrigtsen brothers, cross-country skiing). Two quality sessions per week at exactly lactate threshold (IF 0.85–0.95). Everything else is genuine Z2 (IF < 0.75). Deliberately avoids the "grey zone" (IF 0.75–0.85).

**Why it works:** Most amateur cyclists spend too much time in Z3 (too hard to be recovery, too easy to be threshold stimulus). The Norwegian method forces polarisation — go easy when it's easy, go hard when it's hard.

**Z3 drift — the enemy:** When your "quality" sessions drift to IF 0.78–0.83, you're doing Tempo. You're fatiguing yourself without getting the threshold adaptation. This is what Gemini was doing to you.

**Dashboard monitoring:** The IF distribution histogram should show a bimodal distribution — big spike below 0.75 (Z2 sessions) and smaller spike between 0.85–0.95 (threshold sessions). If it looks like a single hump around 0.78, you're in the grey zone permanently.

---

### Session Quality Score (0–100)
**What it is:** A composite score per session combining three factors:
- W/BPM efficiency vs 28-day baseline (50% weight)
- TSS relative to your average (30% weight)
- IF relative to your average (20% weight)

**How to read it:**
| Score | Meaning |
|---|---|
| 0–30 | Poor session — low quality or high fatigue |
| 30–50 | Below average |
| 50–70 | Normal training session |
| 70–85 | Good quality session |
| 85–100 | Excellent — peak adaptation stimulus |

**Important:** A recovery ride should score low — that's correct. Don't chase high quality scores on easy days. The score is most meaningful for comparing similar session types over time.

---

## 🌡️ ENVIRONMENTAL CORRECTIONS

### Heat Correction (>28°C)
**What it is:** Above 28°C, your HR rises 5-10 bpm for the same power output due to thermoregulation (blood goes to skin to cool you). This makes W/BPM appear lower, which looks like fatigue.

**How the dashboard handles it:** Hot sessions (>28°C) are marked with an orange circle on the efficiency chart so you don't misinterpret heat as fatigue.

**Rule of thumb:** If it's over 30°C, add 8-10 bpm to your expected HR. A session that looks like "bad efficiency" on a 35°C day might actually be perfectly normal.

---

## 📊 HOW TO MONITOR EACH METRIC

### Daily (after each ride)
- TSS — how hard was today?
- IF — was it in the right zone?
- W/BPM vs baseline — objective fatigue check
- Temp — was it hot? Context for efficiency

### Weekly (every Monday)
- TSB — am I fresh or fatigued?
- Weekly TSS — on track with plan?
- Norwegian compliance — 2 quality sessions done?
- Z3 drift sessions — any grey zone contamination?
- CTL trend — is fitness building?

### Monthly
- FTP Stimulus Score by type — is training mix right?
- W/kg trend — are watts per kilo improving?
- Cardiac efficiency trend — is adaptation happening?
- FTP progression — where is estimated FTP?

### Every 6–8 weeks
- Formal FTP test (20-min effort × 0.95)
- Update FTP in Intervals.icu
- Recalculate all zones
- Assess W/kg vs targets

---

## 🎯 YOUR PERSONAL TARGETS

| Metric | Current | 3-month target | Pre-accident |
|---|---|---|---|
| FTP | 235W | 248W | 275W |
| W/kg | 4.12 | 4.35 | 4.82 |
| CTL | 147 | 155+ | ~160 |
| TSB (build) | -18 | -10 to -5 | n/a |
| W/BPM | ~1.25 | 1.30+ | ~1.40 |
| Weekly TSS | 450-600 | 550-650 | 700+ |
| Weekly elevation | 2000m | 3000m | 4000m+ |

---

*Built for: Ruben Fernandez · 57kg · Road cyclist · Castellar del Vallès, Catalonia*
*Data: 2019–2026 · 1030 sessions · Strava + TrainingPeaks + Intervals.icu*
