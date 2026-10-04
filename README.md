# EduGaps-AI — Nigeria Teacher Deployment Urgency Ranker

A machine learning decision-support tool that ranks all 777 Nigerian Local Government Areas (LGAs) by urgency of teacher deployment, using XGBoost residual regression and an AI-powered policy brief generator.

## Live Demo

[https://mikomijie-education-urgency-ranker-app-n0efbq.streamlit.app](https://mikomijie-education-urgency-ranker-app-n0efbq.streamlit.app)

## The Problem

Nigeria publishes aggregate education data — enrollment figures, teacher counts, facility conditions, exam outcomes — but it is scattered across sources and rarely combined in a way that shows where the system is actually failing. Two LGAs can look similar on paper yet have very different learning outcomes, with no simple way for a policymaker to spot the gap and act on it.

## What This Tool Does

EduGaps-AI solves this by asking a different question: **given an LGA's reported resources, how well should it be performing — and how far short is it actually falling?**

The tool builds an XGBoost regression model that predicts each LGA's expected WAEC pass rate purely from its resource inputs. It then computes the residual (Actual − Predicted) for every LGA. Large negative residuals flag hidden systemic failures — LGAs where outcomes fall far below what their resources would predict, pointing to structural problems like ghost teachers, resource diversion, or severe infrastructure deficits.

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Mikomijie/education-urgency-ranker.git
cd education-urgency-ranker
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project root:
OPENROUTER_API_KEY=your_openrouter_api_key_here

Get a free key at [openrouter.ai](https://openrouter.ai)

### 4. Generate the data

```bash
python generate_synthetic_data.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

## System Architecture
[Synthetic Data Engine]
         │
         ▼
[Feature Engineering]
  StudentTeacherRatio
  PupilClassroomRatio
  Classrooms
         │
         ▼
[XGBoost Regressor]
  Predicts expected pass rate
  from resource inputs alone
         │
         ▼
[Residual Engine]
  Residual = Actual minus Predicted
  Z-score standardization
  Urgency tier classification
         │
         ├──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
[Streamlit Dashboard]            [What-If Simulator]
  Top 20 LGA ranker                Adjust teachers,
  State urgency ranking            facility score,
  Feature importance chart         funding per capita
  Scatter plot                     Model updates live
  CSV export
         │
         ▼
[LLM Policy Brief Generator]
  OpenRouter API
  3-bullet intervention plan
  Tailored to each LGA profile
  
## Dashboard Features

**KPI Cards** — Total LGAs analysed, Critical urgency count, Total teachers needed, Average pass rate

**Urgency Distribution** — Bar chart showing how many LGAs fall into each tier (Critical / High / Medium / Low)

**Scatter Plot** — Student-teacher ratio vs actual pass rate, coloured by urgency tier, hover to inspect individual LGAs

**State-Level Urgency Ranking** — All 37 states ranked by average performance gap; red = most underperforming

**Top 20 Priority Table** — Sortable table of the 20 LGAs most urgently needing teacher deployment, with colour-coded residuals

**Feature Importance Chart** — XGBoost's built-in importance scores showing which input features drive underperformance predictions

**What-If Simulator** — Select any LGA, adjust teacher deployment, facility investment and funding, and see the model's predicted pass rate update instantly

**LGA Diagnosis** — Detailed breakdown for any selected LGA including gap size, urgency tier, teachers needed to hit the 1:40 national benchmark

**AI Policy Brief Generator** — One-click button that sends the LGA's profile to an LLM and returns a 3-bullet actionable intervention plan for the State Ministry of Education

**CSV Export** — Download filtered data for offline use

## Urgency Tier Definitions

| Tier | Residual Threshold | Meaning |
|---|---|---|
| Critical | < −15 points | Severe underperformance; immediate intervention required |
| High | < −8 points | Significant gap; priority for next deployment cycle |
| Medium | < 0 points | Below expected; monitor closely |
| Low | ≥ 0 points | Meeting or exceeding predicted performance |

## Model Details

**Algorithm:** XGBRegressor (n_estimators=100, random_state=42)

**Features:**
- `StudentTeacherRatio` — enrollment divided by teacher count
- `Classrooms` — total classroom count per LGA
- `PupilClassroomRatio` — enrollment divided by classrooms

**Target:** `ActualPassRate` (WAEC proxy pass percentage)

**Residual scoring:** `Residual = ActualPassRate − PredictedPassRate`, then Z-score standardised across all 777 LGAs

**Teacher benchmark:** Nigeria's national standard of 1 teacher per 40 pupils (1:40). `TeachersNeeded = (Enrollment / 40) − CurrentTeachers`

## Data

Data is synthetically generated using `generate_synthetic_data.py` to reflect realistic Nigerian education distributions based on known national statistics:

- 777 LGAs across 37 states (including FCT)
- Enrollment ranges: 5,000 to 55,000 pupils per LGA
- Teacher counts: 100 to 1,200 per LGA
- Pass rate ranges: 5% to 98%
- Anomalies injected at realistic rates to simulate underperforming and high-efficiency LGAs

The methodology is designed to apply directly to real EMIS, UBEC, or WAEC data when it becomes available. Synthetic data is clearly labelled throughout the app.

**Real data sources this tool is designed for:**
- UBEC / NBS Education Statistics (LGA-level enrollment and teacher counts)
- WAEC / NECO Public Result Summaries (state and LGA pass rates)
- State Ministry of Education open data

## Project Structure
education-urgency-ranker/
├── app.py # Main Streamlit application
├── generate_synthetic_data.py # Synthetic data generator
├── education_data.csv # Generated dataset (777 LGAs)
├── requirements.txt # Python dependencies
├── .env # API key (not committed — see .gitignore)
├── .gitignore # Excludes .env and cache files
└── README.md # This file

## Requirements
streamlit
pandas
plotly
numpy
xgboost
scikit-learn
scipy
matplotlib
openai
python-dotenv

## Next Steps

- Ingest real UBEC and WAEC data via state ministry APIs
- Add GIS choropleth map using Nigeria's official LGA boundary shapefiles
- Expand LLM brief to include RAG over historical intervention reports
- Build PDF export of policy briefs for direct ministry distribution
- Add user authentication for state-level ministry access

## Track

YDP Datathon 2026 — Track 02: Education