import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('education_data.csv')
X = data[['StudentTeacherRatio', 'Classrooms']]
y = data['ActualPassRate']
print(f"Data prepared - Features shape: {X.shape}, Target Shape : {y.shape}")

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)
print("Model trained successfully!!!")

predicted_pass_rate = model.predict(X)
data['PredictedPassRate'] = predicted_pass_rate
data['Residual'] = data['ActualPassRate'] - data['PredictedPassRate']

top_20_urgent = data.nsmallest(20, 'Residual')[['State', 'LGA', 'Enrollment', 'Teachers', 'StudentTeacherRatio', 'ActualPassRate', 'PredictedPassRate', 'Residual']]

print("\n🚨 TOP 20 LGAs NEEDING URGENT TEACHER DEPLOYMENT:\n")
print(top_20_urgent.to_string(index=False))

import shap
explainer = shap.LinearExplainer(model , X)
shap_values = explainer.shap_values(X)
print("\n SHAP values calculated ! ")