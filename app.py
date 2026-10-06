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

st.set_page_config(page_title="EduGaps-AI", layout="wide")

@st.cache_data
def load_and_train():
    from sklearn.model_selection import cross_val_predict
    df = pd.read_csv('education_data.csv')
    df['PupilClassroomRatio'] = df['Enrollment'] / df['Classrooms']
        df['SchoolType'] = df['SchoolType'].map({'Primary': 0, 'JSS': 1, 'SSS': 2})
    feats = ['StudentTeacherRatio', 'PupilClassroomRatio', 'Classrooms', 'HasElectricity', 'SchoolType']    df['PredictedPassRate'] = np.clip(cross_val_predict(m, df[feats], df['ActualPassRate'], cv=5), 0, 100)
    m.fit(df[feats], df['ActualPassRate'])
    df['Residual'] = df['ActualPassRate'] - df['PredictedPassRate']
    df['ResidualZScore'] = stats.zscore(df['Residual'])
    return df, m, feats

data, model, features = load_and_train()

def get_urgency_tier(z_score):
    if z_score < -1.5:
        return 'Critical'
    elif z_score < -0.5:
        return 'High'
    elif z_score < 0:
        return 'Medium'
    else:
        return 'Low'

data['UrgencyTier'] = data['ResidualZScore'].apply(get_urgency_tier)
data['TeachersNeeded'] = (data['Enrollment'] / 40) - data['Teachers']
data['TeachersNeeded'] = data['TeachersNeeded'].apply(lambda x: max(0, round(x)))

st.sidebar.title("Filters")
state_filter = st.sidebar.multiselect("Filter by State", options=sorted(data['State'].unique()), default=[])
tier_filter = st.sidebar.multiselect("Filter by Urgency Tier", options=['Critical', 'High', 'Medium', 'Low'], default=[])

st.title("EduGaps-AI — Nigeria Education Resource Intelligence")
st.markdown("Ranking all 774 Nigerian LGAsBoost residual regression and AI-powered policy synthesis.")
st.info("""
**Data Note:** This prototype uses synthetically generated data modelled on realistic Nigerian education distributions. 
The methodology (XGBoost residual regression) applies directly to real EMIS/UBEC data when available. 
LGA names are real; figures are illustrative.
""")
filtered_data = data.copy()

if state_filter:
    filtered_data = filtered_data[filtered_data['State'].isin(state_filter)]

if tier_filter:
    filtered_data = filtered_data[filtered_data['UrgencyTier'].isin(tier_filter)]

col1, col2, col3, col4 = st.columns(4)
st.caption("📌 National benchmark: 1 teacher per 40 students (Federal Ministry of Education standard)")
col1.metric("Total LGAs Analysed", len(filtered_data))
col2.metric("Critical Urgency", len(filtered_data[filtered_data['UrgencyTier'] == 'Critical']))
col3.metric("Total Teachers Needed", f"{filtered_data['TeachersNeeded'].sum():,}")
col4.metric("Avg Pass Rate", f"{filtered_data['ActualPassRate'].mean():.1f}%")

color_map = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#ffdd57', 'Low': '#2ca02c'}

st.subheader("Urgency Distribution Across LGAs")
tier_counts = filtered_data['UrgencyTier'].value_counts().reset_index()
tier_counts.columns = ['UrgencyTier', 'Count']
fig1 = px.bar(tier_counts, x='UrgencyTier', y='Count', color='UrgencyTier', color_discrete_map=color_map, title="LGAs by Urgency Tier")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Student-Teacher Ratio vs Pass Rate")
fig2 = px.scatter(
    filtered_data,
    x='StudentTeacherRatio',
    y='ActualPassRate',
    color='UrgencyTier',
    color_discrete_map=color_map,
    hover_data=['LGA', 'State', 'TeachersNeeded'],
    title="Each dot is one LGA"
)
st.plotly_chart(fig2, use_container_width=True)
st.subheader("State-Level Urgency Ranking")
state_summary = filtered_data.groupby('State').agg(
    AvgResidual=('Residual', 'mean'),
    TotalTeachersNeeded=('TeachersNeeded', 'sum'),
    LGACount=('LGA', 'count')
).reset_index().sort_values('AvgResidual')

fig_state = px.bar(
    state_summary,
    x='AvgResidual',
    y='State',
    orientation='h',
    color='AvgResidual',
    color_continuous_scale='RdYlGn',
    hover_data=['TotalTeachersNeeded', 'LGACount'],
    title='Avg. Gap Between Expected & Actual Pass Rate by State (Negative = Underperforming vs. Resources)',
        height=800
)
fig_state.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_state, use_container_width=True)
st.subheader("Top 20 LGAs Requiring Urgent Teacher Deployment")
top_20 = filtered_data.nsmallest(20, 'Residual')[
    ['State', 'LGA', 'Enrollment', 'Teachers', 'StudentTeacherRatio',
     'ActualPassRate', 'PredictedPassRate', 'Residual', 'UrgencyTier', 'TeachersNeeded']
]
st.dataframe(top_20.style.background_gradient(subset=['Residual'], cmap='RdYlGn'), use_container_width=True)

st.subheader("Why Are These LGAs Urgent? (Feature Importance)")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
fig3 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
              title="Feature Importance — XGBoost")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("⭐ Star LGAs — High Efficiency Benchmarks")
st.markdown("These LGAs outperform model predictions the most — study what they're doing right.")
star_lgas = data.nlargest(10, 'Residual')[['LGA', 'State', 'ActualPassRate', 'PredictedPassRate', 'Residual', 'StudentTeacherRatio', 'UrgencyTier']]
st.dataframe(star_lgas.style.background_gradient(subset=['Residual'], cmap='Greens'), use_container_width=True)

st.subheader("Export Data")
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="urgent_lgas.csv",
    mime="text/csv"
)
st.subheader("What-If Simulator")
st.markdown("Adjust resources and see how the predicted pass rate changes.")

critical_lgas = data[data['UrgencyTier'] == 'Critical']['LGA'].tolist()
other_lgas = data[data['UrgencyTier'] != 'Critical']['LGA'].tolist()
sim_lga = st.selectbox("Select LGA to simulate (Critical LGAs listed first)", options=critical_lgas + other_lgas, key='sim_lga')
sim_row = data[data['LGA'] == sim_lga].iloc[0]

col_a, col_b = st.columns(2)
with col_a:
    extra_teachers = st.slider("Extra Teachers Deployed", 0, 200, 0, step=10)
with col_b:
    facility_boost = st.slider("Extra Classrooms Built", 0, 50, 0, step=5)

new_teachers = sim_row['Teachers'] + extra_teachers
new_ratio = sim_row['Enrollment'] / new_teachers

new_classrooms = sim_row['Classrooms'] + facility_boost
new_classrooms = max(new_classrooms, 1)

sim_input = pd.DataFrame([{
    'StudentTeacherRatio': new_ratio,
    'Classrooms': new_classrooms,
    'PupilClassroomRatio': sim_row['Enrollment'] / new_classrooms,
    'HasElectricity': sim_row['HasElectricity'],
    'SchoolType': sim_row['SchoolType']
}])

new_predicted = model.predict(sim_input)[0]
original_predicted = sim_row['PredictedPassRate']
improvement = new_predicted - original_predicted

new_residual = sim_row['ActualPassRate'] - new_predicted
new_zscore = (new_residual - data['Residual'].mean()) / data['Residual'].std()
new_tier = get_urgency_tier(new_zscore)
original_tier = sim_row['UrgencyTier']

col1s, col2s, col3s, col4s = st.columns(4)
col1s.metric("Original Predicted Pass Rate", f"{original_predicted:.1f}%")
col2s.metric("Predicted Pass Rate (after intervention)", f"{new_predicted:.1f}%", delta=f"{improvement:+.1f}%")
col3s.metric("Teachers After Deployment", int(new_teachers))
col4s.metric("Projected Urgency Tier", f"{original_tier} → {new_tier}" if new_tier != original_tier else original_tier)
def generate_policy_brief(row):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    prompt = f"""You are an education policy advisor in Nigeria.
Analyze this LGA and provide exactly 3 bullet points as an action plan:
- LGA: {row['LGA']}, {row['State']}
- Student-Teacher Ratio: 1:{row['StudentTeacherRatio']:.0f}
- Actual Pass Rate: {row['ActualPassRate']:.1f}%
- Expected Pass Rate (ML Model): {row['PredictedPassRate']:.1f}%
- Performance Gap: {abs(round(row['ActualPassRate'] - row['PredictedPassRate'], 1))} points below prediction
- Urgency Tier: {row['UrgencyTier']}
- Teachers Needed (1:40 benchmark): {int(row['TeachersNeeded'])}

Provide 3 specific, actionable bullet points for the State Ministry of Education."""
    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
 
st.subheader("LGA Diagnosis")
selected_lga = st.selectbox("Select an LGA for detailed analysis", options=sorted(data['LGA'].tolist()))

if selected_lga:
    row = data[data['LGA'] == selected_lga].iloc[0]
    gap = round(row['ActualPassRate'] - row['PredictedPassRate'], 1)
    if row['TeachersNeeded'] > 0:
        teachers_line = f"Deploying **{int(row['TeachersNeeded'])} additional teachers** would bring this LGA to the national benchmark."
    else:
        teachers_line = "This LGA already meets the 1:40 teacher benchmark."
    st.info(f"""
**{row['LGA']}, {row['State']}**

This LGA has **{row['StudentTeacherRatio']:.0f} students per teacher** against Nigeria's national average of 1:40.
The model predicted a pass rate of **{row['PredictedPassRate']:.1f}%** given its resources.
Actual pass rate is **{row['ActualPassRate']:.1f}%** — a gap of **{abs(gap)} points**.
Urgency tier: **{row['UrgencyTier']}**.
{teachers_line}
    """)

    if st.button("Generate AI Policy Brief"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            st.warning("OpenRouter API key not configured. Add OPENROUTER_API_KEY to your .env file.")
        else:
            with st.spinner("Generating intervention plan..."):
                brief = generate_policy_brief(row)
            st.subheader("AI-Generated Intervention Plan")
            st.markdown(brief)