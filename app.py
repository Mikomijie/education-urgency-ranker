from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from xgboost import XGBRegressor
from scipy import stats
from sklearn.model_selection import cross_val_predict
import shap
import matplotlib.pyplot as plt

# ── Real NECO 2024 state-level pass rates (Source: Daily Trust, KofaStudy 2024) ──
REAL_NECO = {
    'Abia': 83.4, 'Imo': 81.0, 'Ebonyi': 80.6, 'Cross River': 76.7,
    'Delta': 76.3, 'Ogun': 75.7, 'Ekiti': 75.5, 'Bayelsa': 75.3,
    'Lagos': 73.5, 'Oyo': 74.1, 'Kebbi': 72.5, 'Jigawa': 46.8,
    'Zamfara': 48.7, 'Kano': 44.4, 'Katsina': 42.0, 'Sokoto': 26.8,
    'Adamawa': 51.9, 'Niger': 46.9, 'Anambra': 70.8, 'Osun': 70.7,
}

st.set_page_config(page_title="EduGaps-AI", layout="wide")

@st.cache_data
def load_and_train():
    data = pd.read_csv('education_data.csv')
    data['PupilClassroomRatio'] = data['Enrollment'] / data['Classrooms']

    # Align synthetic state averages to real NECO 2024 figures
    data['HasRealData'] = data['State'].isin(REAL_NECO)
    for state, real_avg in REAL_NECO.items():
        mask = data['State'] == state
        if mask.any():
            synth_avg = data.loc[mask, 'ActualPassRate'].mean()
            shift = real_avg - synth_avg
            data.loc[mask, 'ActualPassRate'] = np.clip(
                data.loc[mask, 'ActualPassRate'] + shift, 5, 98
            )

    features = [
        'StudentTeacherRatio',
        'PupilClassroomRatio',
        'Classrooms',
        'FacilityScore',
        'GradeDropoutRate',
    ]

    model = XGBRegressor(n_estimators=100, random_state=42)
    data['PredictedPassRate'] = cross_val_predict(
        model, data[features], data['ActualPassRate'], cv=5
    )
    data['PredictedPassRate'] = np.clip(data['PredictedPassRate'], 0, 100)
    model.fit(data[features], data['ActualPassRate'])
    data['Residual'] = data['ActualPassRate'] - data['PredictedPassRate']
    data['ResidualZScore'] = stats.zscore(data['Residual'])

    def get_urgency_tier(z):
        if z < -1.5: return 'Critical'
        elif z < -0.5: return 'High'
        elif z < 0: return 'Medium'
        else: return 'Low'

    data['UrgencyTier'] = data['ResidualZScore'].apply(get_urgency_tier)
    data['TeachersNeeded'] = (data['Enrollment'] / 40) - data['Teachers']
    data['TeachersNeeded'] = data['TeachersNeeded'].apply(lambda x: max(0, round(x)))

    # ── Anomaly typing: WHY is this LGA underperforming? ──
    def get_anomaly_type(row):
        high_str = row['StudentTeacherRatio'] > 50
        high_infra = row['PupilClassroomRatio'] > 80
        poor_facility = row['FacilityScore'] < 0.4
        high_dropout = row['GradeDropoutRate'] > 0.25
        underperforming = row['Residual'] < -5

        if not underperforming:
            return 'On Track'
        if high_str and (high_infra or poor_facility):
            return 'Staffing + Infrastructure Gap'
        if high_str:
            return 'Staffing Gap'
        if high_infra or poor_facility:
            return 'Infrastructure Gap'
        if high_dropout:
            return 'High Dropout Risk'
        return 'Unexplained Underperformance'

    data['AnomalyType'] = data.apply(get_anomaly_type, axis=1)

    return data, model, features

data, model, features = load_and_train()

# ── Sidebar filters ──
st.sidebar.title("Filters")
state_filter = st.sidebar.multiselect(
    "Filter by State", options=sorted(data['State'].unique()), default=[]
)
zone_filter = st.sidebar.multiselect(
    "Filter by Geopolitical Zone", options=sorted(data['Zone'].unique()), default=[]
)
tier_filter = st.sidebar.multiselect(
    "Filter by Urgency Tier", options=['Critical', 'High', 'Medium', 'Low'], default=[]
)

# ── Header ──
st.title("EduGaps-AI — Nigeria Education Resource Intelligence")
st.markdown("Ranking all 774 Nigerian LGAs by urgency of educational support using XGBoost residual regression and AI-powered policy synthesis.")

st.info("""
**Data Sources & Context**
- 🏫 **UBEC (2024):** Nigeria needs **194,876 more teachers** and **1,107,854 more classrooms** nationally
- 📊 **NECO 2024:** State-level pass rates from real results — Abia leads at 83.4%, Sokoto trails at 26.8%
- 👻 **Teacher absenteeism:** ~25% nationally (World Metrics / Ministry of Education)
- 📁 State pass rates use real NECO 2024 data for 20 states; LGA figures are synthetic modelled on UBEC/NBS distributions
""")

filtered_data = data.copy()
if state_filter:
    filtered_data = filtered_data[filtered_data['State'].isin(state_filter)]
if zone_filter:
    filtered_data = filtered_data[filtered_data['Zone'].isin(zone_filter)]
if tier_filter:
    filtered_data = filtered_data[filtered_data['UrgencyTier'].isin(tier_filter)]

# ── KPI metrics ──
from sklearn.metrics import r2_score, mean_squared_error

col1, col2, col3, col4, col5 = st.columns(5)
st.caption("📌 National benchmark: 1 teacher per 40 students (Federal Ministry of Education standard)")
col1.metric("Total LGAs Analysed", len(filtered_data))
col2.metric("Critical Urgency", len(filtered_data[filtered_data['UrgencyTier'] == 'Critical']))
col3.metric("Total Teachers Needed", f"{filtered_data['TeachersNeeded'].sum():,}")
col4.metric("Avg Pass Rate", f"{filtered_data['ActualPassRate'].mean():.1f}%")

r2 = r2_score(data['ActualPassRate'], data['PredictedPassRate'])
rmse = np.sqrt(mean_squared_error(data['ActualPassRate'], data['PredictedPassRate']))
col5.metric("Model R²", f"{r2:.2f}", help=f"RMSE: {rmse:.1f} pts — how well XGBoost predicts pass rate from resources")
critical_df = data[data['UrgencyTier'] == 'Critical']
teachers_to_deploy = int(critical_df['TeachersNeeded'].sum())
students_served = int(critical_df[critical_df['TeachersNeeded'] > 0]['Enrollment'].sum())
st.success(f"🎯 Deploying **{teachers_to_deploy:,} teachers** to Critical-tier LGAs would bring **{students_served:,} students** to Nigeria's 1:40 federal benchmark.")
color_map = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#ffdd57', 'Low': '#2ca02c'}

# ── Nigeria State Map ──
st.subheader("🗺️ Nigeria: Education Underperformance by State")
st.caption("Bubble size = teachers needed. Colour = performance gap vs. resources. Red = underperforming, Green = overperforming.")

state_summary = filtered_data.groupby(['State', 'Zone']).agg(
    AvgResidual=('Residual', 'mean'),
    AvgPassRate=('ActualPassRate', 'mean'),
    TotalTeachersNeeded=('TeachersNeeded', 'sum'),
    LGACount=('LGA', 'count')
).reset_index()

state_summary['RealNECO'] = state_summary['State'].map(REAL_NECO)
state_summary['DataSource'] = state_summary['RealNECO'].apply(
    lambda x: 'Real NECO 2024' if pd.notna(x) else 'Synthetic (UBEC-modelled)'
)

STATE_COORDS = {
    'Abia': (5.45, 7.52), 'Adamawa': (9.33, 12.40), 'Akwa Ibom': (5.01, 7.92),
    'Anambra': (6.21, 7.07), 'Bauchi': (10.31, 9.84), 'Bayelsa': (4.77, 6.07),
    'Benue': (7.34, 8.74), 'Borno': (11.85, 13.16), 'Cross River': (5.87, 8.60),
    'Delta': (5.68, 5.95), 'Ebonyi': (6.26, 8.01), 'Edo': (6.34, 5.63),
    'Ekiti': (7.62, 5.23), 'Enugu': (6.46, 7.55), 'FCT': (8.90, 7.38),
    'Gombe': (10.29, 11.17), 'Imo': (5.57, 7.06), 'Jigawa': (12.23, 9.56),
    'Kaduna': (10.52, 7.44), 'Kano': (12.00, 8.52), 'Katsina': (12.99, 7.62),
    'Kebbi': (11.49, 4.20), 'Kogi': (7.73, 6.69), 'Kwara': (8.97, 4.39),
    'Lagos': (6.52, 3.38), 'Nasarawa': (8.49, 8.20), 'Niger': (9.93, 5.60),
    'Ogun': (6.99, 3.47), 'Ondo': (7.09, 5.20), 'Osun': (7.56, 4.56),
    'Oyo': (7.85, 3.93), 'Plateau': (9.22, 9.52), 'Rivers': (4.84, 6.91),
    'Sokoto': (13.06, 5.24), 'Taraba': (7.87, 11.36), 'Yobe': (12.29, 11.44),
    'Zamfara': (12.17, 6.22),
}
state_summary['Lat'] = state_summary['State'].map(lambda s: STATE_COORDS.get(s, (9.0, 8.0))[0])
state_summary['Lon'] = state_summary['State'].map(lambda s: STATE_COORDS.get(s, (9.0, 8.0))[1])

fig_map = px.scatter_geo(
    state_summary,
    lat='Lat', lon='Lon',
    color='AvgResidual',
    size=state_summary['TotalTeachersNeeded'].clip(lower=100),
    hover_name='State',
    hover_data={
        'Zone': True,
        'AvgResidual': ':.1f',
        'RealNECO': ':.1f',
        'TotalTeachersNeeded': ':,',
        'DataSource': True,
        'Lat': False, 'Lon': False,
    },
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0,
    size_max=40,
    scope='africa',
    title='State Underperformance Gap (red = below resources, green = above)',
)
fig_map.update_geos(
    center=dict(lat=9.0, lon=8.0),
    projection_scale=5,
    showland=True, landcolor='#f0f0f0',
    showocean=True, oceancolor='#d0e8f0',
    showcoastlines=True, coastlinecolor='#aaa',
    showcountries=True, countrycolor='#888',
)
fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_map, use_container_width=True)

# ── Zone-level summary ──
st.subheader("📍 Geopolitical Zone Summary")
zone_summary = filtered_data.groupby('Zone').agg(
    AvgResidual=('Residual', 'mean'),
    AvgPassRate=('ActualPassRate', 'mean'),
    TotalTeachersNeeded=('TeachersNeeded', 'sum'),
    AvgFacilityScore=('FacilityScore', 'mean'),
    AvgDropoutRate=('GradeDropoutRate', 'mean'),
    CriticalLGAs=('UrgencyTier', lambda x: (x == 'Critical').sum()),
).reset_index().sort_values('AvgResidual')
st.dataframe(zone_summary.style.background_gradient(subset=['AvgResidual'], cmap='RdYlGn'), use_container_width=True)

# ── Urgency distribution ──
st.subheader("Urgency Distribution Across LGAs")
tier_counts = filtered_data['UrgencyTier'].value_counts().reset_index()
tier_counts.columns = ['UrgencyTier', 'Count']
fig1 = px.bar(tier_counts, x='UrgencyTier', y='Count', color='UrgencyTier',
              color_discrete_map=color_map, title="LGAs by Urgency Tier")
st.plotly_chart(fig1, use_container_width=True)

# ── Scatter ──
st.subheader("Student-Teacher Ratio vs Pass Rate")
fig2 = px.scatter(
    filtered_data, x='StudentTeacherRatio', y='ActualPassRate',
    color='UrgencyTier', color_discrete_map=color_map,
    hover_data=['LGA', 'State', 'Zone', 'TeachersNeeded', 'AnomalyType', 'FacilityScore'],
    title="Each dot is one LGA — hover for details"
)
st.plotly_chart(fig2, use_container_width=True)

# ── State bar chart ──
st.subheader("State-Level Urgency Ranking")
fig_state = px.bar(
    state_summary.sort_values('AvgResidual'),
    x='AvgResidual', y='State', orientation='h',
    color='AvgResidual', color_continuous_scale='RdYlGn',
    hover_data=['TotalTeachersNeeded', 'LGACount', 'DataSource'],
    title='Avg. Gap Between Expected & Actual Pass Rate by State',
    height=800
)
fig_state.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_state, use_container_width=True)

# ── Top 20 table ──
st.subheader("Top 20 LGAs Requiring Urgent Intervention")
top_20 = filtered_data.nsmallest(20, 'Residual')[
    ['State', 'Zone', 'LGA', 'Enrollment', 'Teachers', 'StudentTeacherRatio',
     'FacilityScore', 'GradeDropoutRate', 'ActualPassRate', 'PredictedPassRate',
     'Residual', 'UrgencyTier', 'TeachersNeeded', 'AnomalyType']
]
st.dataframe(top_20.style.background_gradient(subset=['Residual'], cmap='RdYlGn'), use_container_width=True)

# ── Anomaly breakdown ──
st.subheader("🔍 Why Are LGAs Underperforming?")
anomaly_counts = filtered_data[
    filtered_data['UrgencyTier'].isin(['Critical', 'High'])
]['AnomalyType'].value_counts().reset_index()
anomaly_counts.columns = ['AnomalyType', 'Count']
anomaly_color = {
    'Staffing Gap': '#d62728',
    'Infrastructure Gap': '#ff7f0e',
    'Staffing + Infrastructure Gap': '#9467bd',
    'High Dropout Risk': '#e377c2',
    'Unexplained Underperformance': '#8c564b',
    'On Track': '#2ca02c',
}
fig_anomaly = px.bar(
    anomaly_counts, x='AnomalyType', y='Count',
    color='AnomalyType', color_discrete_map=anomaly_color,
    title="Root Cause of Underperformance (Critical + High urgency LGAs)"
)
st.plotly_chart(fig_anomaly, use_container_width=True)

# ── Unexplained underperformance callout ──
unexplained = filtered_data[
    (filtered_data['AnomalyType'] == 'Unexplained Underperformance') &
    (filtered_data['UrgencyTier'].isin(['Critical', 'High']))
]
if len(unexplained) > 0:
    st.error(f"""
**⚠️ {len(unexplained)} LGAs show Unexplained Underperformance**

These LGAs have adequate teachers and classrooms on paper — but their pass rates are still far below what the model predicts.
This is not a deployment problem. It is a **governance and accountability problem** — likely caused by ghost teachers, chronic absenteeism (~25% nationally), or diverted resources.

**SUBEB recommendation:** Prioritise these LGAs for audit and accountability review, not additional staffing.
""")
    st.dataframe(
        unexplained[['State', 'Zone', 'LGA', 'StudentTeacherRatio', 'FacilityScore',
                      'ActualPassRate', 'PredictedPassRate', 'Residual']
        ].sort_values('Residual'),
        use_container_width=True
    )

# ── Feature importance ──
st.subheader("Feature Importance — What Drives Pass Rate Predictions?")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)
fig3 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
              title="XGBoost Feature Importance")
st.plotly_chart(fig3, use_container_width=True)

# ── Star LGAs ──
st.subheader("⭐ Star LGAs — High Efficiency Benchmarks")
st.markdown("These LGAs outperform model predictions the most — study what they're doing right.")
star_lgas = data.nlargest(10, 'Residual')[
    ['LGA', 'State', 'Zone', 'ActualPassRate', 'PredictedPassRate',
     'Residual', 'StudentTeacherRatio', 'FacilityScore', 'UrgencyTier']
]
st.dataframe(star_lgas.style.background_gradient(subset=['Residual'], cmap='Greens'), use_container_width=True)

# ── Export ──
st.subheader("Export Data")
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Filtered Data as CSV", data=csv,
                   file_name="urgent_lgas.csv", mime="text/csv")

# ── What-If Simulator ──
st.subheader("What-If Simulator")
st.markdown("Select a Critical LGA, adjust resources, and see the predicted impact.")
st.caption("Built for SUBEB officers making teacher deployment decisions.")

critical_lgas = data[data['UrgencyTier'] == 'Critical']['LGA'].tolist()
other_lgas = data[data['UrgencyTier'] != 'Critical']['LGA'].tolist()
sim_lga = st.selectbox(
    "Select LGA to simulate (Critical LGAs listed first)",
    options=critical_lgas + other_lgas, key='sim_lga'
)
sim_row = data[data['LGA'] == sim_lga].iloc[0]

col_a, col_b = st.columns(2)
with col_a:
    extra_teachers = st.slider("Extra Teachers Deployed", 0, 200, 0, step=10)
with col_b:
    facility_boost = st.slider("Extra Classrooms Built", 0, 50, 0, step=5)

new_teachers = sim_row['Teachers'] + extra_teachers
new_ratio = sim_row['Enrollment'] / new_teachers
new_classrooms = max(sim_row['Classrooms'] + facility_boost, 1)

sim_input = pd.DataFrame([{
    'StudentTeacherRatio': new_ratio,
    'Classrooms': new_classrooms,
    'PupilClassroomRatio': sim_row['Enrollment'] / new_classrooms,
    'FacilityScore': sim_row['FacilityScore'],
    'GradeDropoutRate': sim_row['GradeDropoutRate'],
}])[features]

new_predicted = model.predict(sim_input)[0]
original_predicted = sim_row['PredictedPassRate']
improvement = new_predicted - original_predicted

new_residual = sim_row['ActualPassRate'] - new_predicted
new_zscore = (new_residual - data['Residual'].mean()) / data['Residual'].std()

def get_urgency_tier(z):
    if z < -1.5: return 'Critical'
    elif z < -0.5: return 'High'
    elif z < 0: return 'Medium'
    else: return 'Low'

new_tier = get_urgency_tier(new_zscore)
original_tier = sim_row['UrgencyTier']

col1s, col2s, col3s, col4s = st.columns(4)
col1s.metric("Original Predicted Pass Rate", f"{original_predicted:.1f}%")
col2s.metric("Predicted Pass Rate (after intervention)", f"{new_predicted:.1f}%", delta=f"{improvement:+.1f}%")
col3s.metric("Teachers After Deployment", int(new_teachers))
col4s.metric("Urgency Tier Change", f"{original_tier} → {new_tier}")

# ── AI Policy Brief ──
def generate_policy_brief(row):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    prompt = f"""You are an education policy advisor writing for Nigeria's State Universal Basic Education Board (SUBEB).
Analyze this LGA and provide exactly 3 bullet points as a concrete action plan:
- LGA: {row['LGA']}, {row['State']} ({row['Zone']})
- Student-Teacher Ratio: 1:{row['StudentTeacherRatio']:.0f}
- Facility Score: {row['FacilityScore']:.2f} out of 1.0
- Grade Dropout Rate: {row['GradeDropoutRate']*100:.1f}%
- Actual Pass Rate: {row['ActualPassRate']:.1f}%
- Expected Pass Rate (ML Model): {row['PredictedPassRate']:.1f}%
- Performance Gap: {abs(round(row['ActualPassRate'] - row['PredictedPassRate'], 1))} points below prediction
- Urgency Tier: {row['UrgencyTier']}
- Root Cause: {row['AnomalyType']}
- Teachers Needed (1:40 benchmark): {int(row['TeachersNeeded'])}

Provide 3 specific, actionable bullet points for the SUBEB officer making deployment decisions."""
    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── LGA Diagnosis ──
st.subheader("LGA Diagnosis")
selected_lga = st.selectbox("Select an LGA for detailed analysis", options=top_20['LGA'].tolist())

if selected_lga:
    row = data[data['LGA'] == selected_lga].iloc[0]
    gap = round(row['ActualPassRate'] - row['PredictedPassRate'], 1)
    teachers_line = (
        f"Deploying **{int(row['TeachersNeeded'])} additional teachers** would bring this LGA to the national benchmark."
        if row['TeachersNeeded'] > 0
        else "This LGA already meets the 1:40 teacher benchmark."
    )
    st.info(f"""
**{row['LGA']}, {row['State']} — {row['Zone']}**

This LGA has **{row['StudentTeacherRatio']:.0f} students per teacher** against the national benchmark of 1:40.
Facility score: **{row['FacilityScore']:.2f}/1.0** | Grade dropout rate: **{row['GradeDropoutRate']*100:.1f}%**
The model predicted a pass rate of **{row['PredictedPassRate']:.1f}%** given its resources.
Actual pass rate is **{row['ActualPassRate']:.1f}%** — a gap of **{abs(gap)} points**.
Urgency tier: **{row['UrgencyTier']}** | Root cause: **{row['AnomalyType']}**
{teachers_line}
    """)

    # ── SHAP waterfall chart ──
    st.markdown("**Why did the model flag this LGA? (SHAP Explanation)**")
    explainer = shap.TreeExplainer(model)
    row_input = pd.DataFrame([{
        'StudentTeacherRatio': row['StudentTeacherRatio'],
        'PupilClassroomRatio': row['PupilClassroomRatio'],
        'Classrooms': row['Classrooms'],
        'FacilityScore': row['FacilityScore'],
        'GradeDropoutRate': row['GradeDropoutRate'],
    }])
    shap_values = explainer(row_input)
    fig_shap, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig_shap, bbox_inches='tight')
    plt.close()

    if st.button("Generate AI Policy Brief"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            st.warning("OpenRouter API key not configured. Add OPENROUTER_API_KEY to your .env file.")
        else:
            with st.spinner("Generating SUBEB intervention plan..."):
                brief = generate_policy_brief(row)
            st.subheader("AI-Generated Intervention Plan")
            st.markdown(brief)