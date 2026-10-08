# EduGaps-AI — Nigeria Education Resource Intelligence

> Ranking all 777 Nigerian LGAs by urgency of teacher deployment using XGBoost residual regression, SHAP explainability, and AI-powered policy synthesis.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![XGBoost](https://img.shields.io/badge/XGBoost-ML-green) ![Track](https://img.shields.io/badge/Track-Education-orange)

---

## The Problem

Nigeria publishes education data — enrollment figures, teacher numbers, facility conditions, exam outcomes — but it's scattered across UBEC, state ministries, WAEC/NECO, and NBS. No single tool brings it together to show, at a glance, where the system is actually failing.

Two LGAs can look identical on paper — same enrollment, same reported funding — yet produce wildly different learning outcomes. A policymaker today has no simple way to spot that gap and act on it early.

**EduGaps-AI solves this.** It doesn't just show who has fewer teachers. It shows who is underperforming *relative to the resources they already have* — and that distinction changes everything about how you intervene.

---

## How It Works

### The Core Model: Expected vs. Actual

We train an XGBoost regression model on five features per LGA:

| Feature | What it captures |
|---|---|
| `StudentTeacherRatio` | Teacher availability |
| `PupilClassroomRatio` | Physical space per student |
| `Classrooms` | Raw infrastructure count |
| `FacilityScore` | Overall school condition (0–1) |
| `GradeDropoutRate` | Fraction of pupils who drop before JSS3 |

The model predicts what a LGA's **exam pass rate should be** given its resources. We then compute the **residual** (Actual − Predicted). A large negative residual means the LGA is performing far below what its resources would predict — and that's the real signal.

### Urgency Tiers

Each LGA is assigned an urgency tier based on its residual Z-score:

| Tier | Z-score | Meaning |
|---|---|---|
| 🔴 Critical | < −1.5 | Severely underperforming vs. resources |
| 🟠 High | −1.5 to −0.5 | Meaningfully behind predictions |
| 🟡 Medium | −0.5 to 0 | Slightly below expectations |
| 🟢 Low | ≥ 0 | Meeting or exceeding predicted outcomes |

### Anomaly Typing

Not all underperformance has the same root cause. EduGaps-AI classifies each flagged LGA into one of five anomaly types:

- **Staffing Gap** — teacher ratio is the dominant problem
- **Infrastructure Gap** — classrooms and facility score are the bottleneck
- **Staffing + Infrastructure Gap** — both are failing simultaneously
- **High Dropout Risk** — dropout rate is the primary driver of underperformance
- **Unexplained Underperformance** — resources are adequate but results are still poor (governance/accountability issue — flag for SUBEB audit)

This matters because deploying teachers to an "Unexplained Underperformance" LGA won't help. That LGA needs an audit, not a deployment.

---

## Features

- **National map** — Nigeria scatter_geo bubble map coloured by urgency tier, sized by enrollment
- **State-level ranking** — horizontal bar chart showing average resource-outcome gap per state
- **Top 20 LGA table** — sortable, colour-coded, with teachers needed to reach the 1:40 federal benchmark
- **Zone-level summary** — aggregated view across Nigeria's 6 geopolitical zones
- **⭐ Star LGAs** — LGAs outperforming predictions the most (benchmarks for what good looks like)
- **What-If Simulator** — adjust teacher deployment and classroom construction, see the predicted pass rate change and tier shift in real time
- **SHAP waterfall chart** — per-LGA breakdown of exactly which features are driving the model's prediction (no black box)
- **LGA Diagnosis** — plain-language summary of each LGA's situation and what it would take to reach benchmark
- **AI Policy Brief** — one-click generation of a 3-point intervention plan for the State Ministry of Education, powered by OpenRouter
- **CSV export** — download the full filtered dataset for offline use

---

## Data

### Synthetic Data (Current Prototype)

This prototype uses synthetically generated data modelled on realistic Nigerian education distributions. LGA names are real (all 777 LGAs across 36 states + FCT). Figures are illustrative.

The synthetic data is calibrated to:
- **Real NECO 2024 state-level pass rates** for 20 states (Kano, Lagos, Ogun, Rivers, Kaduna, etc.)
- **UBEC national benchmarks**: 194,876 teachers needed nationally; 1,107,854 classrooms needed
- **Zone-aware profiles**: North West and North East zones modelled with higher dropout and weaker facility scores; South West and South East with stronger baselines — reflecting documented regional disparities
- **Teacher absenteeism**: ~25% national average factored into effective teacher counts

### Real Data (Next Step)

The methodology applies directly to real EMIS/UBEC data. Swap `education_data.csv` for real data with the same column schema and the entire pipeline runs unchanged.

Suggested real data sources:
- UBEC Education Data (ubec.gov.ng)
- NBS Education Statistics
- WAEC/NECO public result summaries
- State ministry of education open datasets

---

## Geopolitical Zone Profiles

| Zone | States | Baseline Profile |
|---|---|---|
| North West | Kano, Kaduna, Sokoto, Zamfara, Kebbi, Katsina, Jigawa | Weakest — highest dropout, lowest facility score |
| North East | Borno, Yobe, Gombe, Adamawa, Taraba, Bauchi | Very weak — conflict impact on infrastructure |
| North Central | Niger, Kogi, Benue, Plateau, Nassarawa, Kwara, FCT | Moderate |
| South West | Lagos, Ogun, Oyo, Osun, Ondo, Ekiti | Strongest — best facility scores, lowest dropout |
| South East | Anambra, Imo, Enugu, Ebonyi, Abia | Strong |
| South South | Rivers, Delta, Edo, Akwa Ibom, Cross River, Bayelsa | Moderate-strong |

---

## Installation

```bash
git clone https://github.com/Mikomijie/education-urgency-ranker.git
cd education-urgency-ranker

pip install -r requirements.txt

# Optional: regenerate the synthetic dataset
python generate_synthetic_data.py

# Run the app
streamlit run app.py
```

### Environment Variables

Create a `.env` file in the project root:
OPENROUTER_API_KEY=your_key_here

The AI Policy Brief feature requires an OpenRouter API key. All other features work without it.

---

## Requirements
streamlit
pandas
numpy
plotly
xgboost
scikit-learn
scipy
shap
openai
python-dotenv

---

## Judging Rubric Alignment

| Criterion | How EduGaps-AI addresses it |
|---|---|
| **Problem relevance & clarity** | Sharply scoped: "which LGAs are underperforming relative to their resources, and why" — not just "AI for education" |
| **Technical execution** | XGBoost residual regression + SHAP explainability + anomaly typing; runs on real-labelled synthetic data; handles all 777 LGAs |
| **Impact potential** | Anomaly typing means interventions are targeted, not generic; Unexplained Underperformance flag creates a direct path to SUBEB audit referral |
| **Presentation & clarity** | Non-technical judges can read urgency tiers and the LGA Diagnosis section without knowing what XGBoost is |
| **Creativity** | Residual-based scoring (not raw ratios) + anomaly typing as a policy routing mechanism is a fresh framing |

---

## What's Working vs. What's Next

### Working now
- Full pipeline: synthetic data → model → dashboard → AI brief
- All 777 LGAs scored, tiered, and anomaly-typed
- SHAP explainability per LGA
- What-If Simulator with tier-shift indicator
- Zone and state filters

### Placeholders
- Data is synthetic (real UBEC/NECO data would plug straight in)
- Map uses state centroids, not true LGA polygons
- AI brief uses OpenRouter auto-routing (not a fine-tuned model)

### What we'd build next
- Integrate real EMIS/UBEC data via API
- True LGA-level choropleth with GADM shapefiles
- Time-series tracking (flag LGAs getting worse year-on-year)
- SUBEB referral workflow integration
- Mobile-friendly field officer view

---

## Team

Built by **Mikomijie** for the YDP Community Datathon 2026 — Education Track.

---

## License

MIT