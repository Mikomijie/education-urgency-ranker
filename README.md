# EduGaps-AI — Nigeria Education Resource Intelligence

> Ranking all 774 Nigerian LGAs by urgency of teacher deployment using XGBoost residual regression and AI-powered policy synthesis.

🔗 **Live Demo:** https://mikomijie-education-urgency-ranker-app-n0efbq.streamlit.app/ 
📁 **GitHub:** https://github.com/Mikomijie/education-urgency-ranker

---

## What Problem Does This Solve?

Nigeria has 774 Local Government Areas, each with schools that vary wildly in teacher availability, classroom capacity, and exam outcomes. The Federal Ministry of Education cannot deploy teachers everywhere at once — they need to know **where intervention is most urgent**.

The standard approach is to rank LGAs by raw pass rate or teacher count. But this is misleading: an LGA with few teachers but good outcomes doesn't need urgent help, while an LGA with adequate resources but terrible outcomes is a crisis hiding in plain sight.

**EduGaps-AI solves this differently.** It asks: *given the resources an LGA has, what pass rate should it be achieving?* LGAs that fall far below their expected performance are flagged as urgent — because the problem there is systemic, not just a matter of adding more resources.

---

## How It Works — The ML Methodology

### Step 1: Learn What "Expected" Performance Looks Like
An XGBoost regression model is trained on 5 features per LGA:
- `StudentTeacherRatio` — students per teacher
- `PupilClassroomRatio` — students per classroom
- `Classrooms` — total classroom count
- `HasElectricity` — whether the school zone has electricity (0/1)
- `SchoolType` — encoded: Primary=0, JSS=1, SSS=2

The model learns what pass rate a given set of resources *should* produce, using **5-fold cross-validation with out-of-fold predictions** to prevent data leakage. Residuals are computed on held-out folds only.

### Step 2: Compute the Performance Gap (Residual)
Residual = ActualPassRate − PredictedPassRate
A **negative residual** means the LGA is underperforming vs. its resources — a red flag.

### Step 3: Z-Score Urgency Tiering
Residuals are standardised into Z-scores and bucketed into 4 tiers:

| Tier | Z-Score | Meaning |
|------|---------|---------|
| 🔴 Critical | < −1.5 | Severely underperforming — immediate intervention needed |
| 🟠 High | < −0.5 | Underperforming — high priority |
| 🟡 Medium | < 0 | Slightly below expectation |
| 🟢 Low | ≥ 0 | Meeting or exceeding expectations |

---

## Features

### 📊 National Dashboard
- KPI cards: Total LGAs, Critical count, Teachers Needed, Avg Pass Rate
- Urgency distribution bar chart
- Student-Teacher Ratio vs Pass Rate scatter plot (coloured by urgency tier)
- State-level average performance gap ranked chart
- Top 20 most urgent LGAs table with colour gradient

### ⭐ Star LGAs — Efficiency Benchmarks
Top 10 LGAs that outperform their predicted pass rate the most. These are high-efficiency outliers — studying what they do differently is as valuable as fixing the worst performers.

### 🔬 What-If Simulator
Select any LGA and adjust:
- **Extra Teachers Deployed** (0–200)
- **Extra Classrooms Built** (0–50)

The model re-predicts in real time, showing the new expected pass rate and projected urgency tier change. Built for policymakers to run scenarios before committing resources.

### 🩺 LGA Diagnosis
Select any of the 774 LGAs for a plain-language breakdown:
- Current student-teacher ratio vs national 1:40 benchmark
- Actual vs predicted pass rate gap
- Exact number of teachers needed to reach benchmark

### 🤖 AI Policy Brief Generator
Powered by OpenRouter LLM. For any selected LGA, generates 3 specific, actionable intervention recommendations for the State Ministry of Education — tailored to that LGA's exact data profile.

### 📥 Export
Download filtered LGA data as CSV for offline analysis.

---

## Data

This prototype uses **synthetically generated data** modelled on realistic Nigerian education distributions. Real LGA names are used (all 774 across 36 states + FCT). Figures are illustrative.

The synthetic data generator (`generate_synthetic_data.py`) produces:
- Realistic resource distributions (enrollment, teachers, classrooms)
- Pass rates with meaningful signal from resources plus real-world noise
- Anomaly injection: 8% severe underperformers (ghost teachers, resource diversion), 18% mild underperformers, 12% high-efficiency star LGAs
- School type distribution: 50% Primary, 30% JSS, 20% SSS
- Electricity access: 60% have electricity

The methodology applies directly to real EMIS/UBEC data when available.

**Limitations & Real-World Validation Path:**
On real EMIS data, we'd expect missing values in teacher counts, inconsistent LGA name spellings across datasets, and enrollment figures that don't match exam cohorts. The preprocessing pipeline would need to handle these before residuals are computed. The model architecture remains unchanged — only the input data changes. A pilot with one state's real data (e.g. Lagos or Kano SUBEB records) would be the natural next validation step.
---

## Project Structure
education-urgency-ranker/
├── app.py                      # Main Streamlit application
├── generate_synthetic_data.py  # Data generation script
├── education_data.csv          # Generated dataset (774 LGAs)
├── requirements.txt            # Python dependencies
└── README.md

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost (XGBRegressor) |
| Validation | 5-fold cross-validation (sklearn) |
| Dashboard | Streamlit |
| Charts | Plotly Express |
| AI Briefs | OpenRouter API (LLM) |
| Data | Pandas, NumPy |
| Statistics | SciPy (Z-score) |

---

## Running Locally

```bash
git clone https://github.com/Mikomijie/education-urgency-ranker
cd education-urgency-ranker
pip install -r requirements.txt

# Add your OpenRouter API key
echo "OPENROUTER_API_KEY=your-key-here" > .env

streamlit run app.py
```

---

## National Benchmark

All teacher deployment calculations use the **Federal Ministry of Education standard of 1 teacher per 40 students** as the national benchmark. `TeachersNeeded = (Enrollment ÷ 40) − CurrentTeachers`

---

## Datathon Context

Built for the **YDP Datathon Nigeria — Education Track (Track 02)**.  
The residual regression approach was chosen specifically because it identifies *systemic underperformance* rather than just resource poverty — an important distinction for policy targeting.

> *"Rather than ranking LGAs by raw pass rate, EduGaps-AI uses XGBoost to learn what pass rate a given set of resources should produce. LGAs where actual results fall significantly below this prediction (negative residual, Z-score < −1.5) are flagged as Critical — these are places where resources exist but outcomes still disappoint, suggesting systemic issues beyond mere underfunding."*