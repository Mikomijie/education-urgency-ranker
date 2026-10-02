import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import shap

st.set_page_config(page_title="Nigeria Teacher Deployment Urgency", layout="wide")

data = pd.read_csv('education_data.csv')

model = LinearRegression()
model.fit(data[['StudentTeacherRatio', 'Classrooms']], data['ActualPassRate'])
data['PredictedPassRate'] = model.predict(data[['StudentTeacherRatio', 'Classrooms']])
data['Residual'] = data['ActualPassRate'] - data['PredictedPassRate']

def get_urgency_tier(residual):
    if residual < -15:
        return 'Critical'
    elif residual < -8:
        return 'High'
    elif residual < 0:
        return 'Medium'
    else:
        return 'Low'

data['UrgencyTier'] = data['Residual'].apply(get_urgency_tier)
data['TeachersNeeded'] = (data['Enrollment'] / 40) - data['Teachers']
data['TeachersNeeded'] = data['TeachersNeeded'].apply(lambda x: max(0, round(x)))

st.sidebar.title("Filters")
state_filter = st.sidebar.multiselect("Filter by State", options=sorted(data['State'].unique()), default=[])
tier_filter = st.sidebar.multiselect("Filter by Urgency Tier", options=['Critical', 'High', 'Medium', 'Low'], default=[])

st.title("Nigeria Teacher Deployment Urgency Dashboard")
st.markdown("Ranking 777 LGAs by urgency of teacher deployment using residual regression and SHAP analysis.")
st.caption("Note: Data is synthetically generated to reflect realistic Nigerian education distributions. Methodology applies directly to real EMIS data.")

filtered_data = data.copy()

if state_filter:
    filtered_data = filtered_data[filtered_data['State'].isin(state_filter)]

if tier_filter:
    filtered_data = filtered_data[filtered_data['UrgencyTier'].isin(tier_filter)]

col1, col2, col3, col4 = st.columns(4)
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

st.subheader("Top 20 LGAs Requiring Urgent Teacher Deployment")
top_20 = filtered_data.nsmallest(20, 'Residual')[
    ['State', 'LGA', 'Enrollment', 'Teachers', 'StudentTeacherRatio',
     'ActualPassRate', 'PredictedPassRate', 'Residual', 'UrgencyTier', 'TeachersNeeded']
]
st.dataframe(top_20.style.background_gradient(subset=['Residual'], cmap='RdYlGn'), use_container_width=True)

st.subheader("Why Are These LGAs Urgent? (SHAP Explanation)")
explainer = shap.LinearExplainer(model, data[['StudentTeacherRatio', 'Classrooms']])
shap_values = explainer.shap_values(data[['StudentTeacherRatio', 'Classrooms']])
shap_df = pd.DataFrame({'Feature': ['StudentTeacherRatio', 'Classrooms'],
                         'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)})
fig3 = px.bar(shap_df, x='MeanAbsSHAP', y='Feature', orientation='h', title="Feature Importance via SHAP")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Export Data")
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="urgent_lgas.csv",
    mime="text/csv"
)
st.subheader("LGA Diagnosis")

selected_lga = st.selectbox("Select an LGA for detailed analysis", options=top_20['LGA'].tolist())

if selected_lga:
    row = data[data['LGA'] == selected_lga].iloc[0]
    gap = round(row['ActualPassRate'] - row['PredictedPassRate'], 1)
    if row['TeachersNeeded'] > 0:
        teachers_line = f"Deploying **{int(row['TeachersNeeded'])} additional teachers** would bring this LGA to the national benchmark."
    else:
        teachers_line = "This LGA already meets the 1:40 teacher benchmark."
    st.info(f"""
**{row['LGA']}, {row['State']}**

This LGA has **{row['StudentTeacherRatio']:.0f} students per teacher**  against Nigeria's national average of 1:40.
The model predicted a pass rate of **{row['PredictedPassRate']:.1f}%** given its resources.
Actual pass rate is **{row['ActualPassRate']:.1f}%**  a gap of **{abs(gap)} points**.
Urgency tier: **{row['UrgencyTier']}**.
{teachers_line}
    """)