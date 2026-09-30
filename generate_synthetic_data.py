import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# List of Nigerian states
states = [
    'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa',
    'Benue', 'Borno', 'Cross River', 'Delta', 'Ebonyi', 'Edo',
    'Ekiti', 'Enugu', 'Gombe', 'Imo', 'Jigawa', 'Kaduna',
    'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos',
    'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo',
    'Plateau', 'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara', 'FCT'
]

# Create LGAs
lgas_per_state = 21
lga_data = []

for state in states:
    for i in range(lgas_per_state):
        lga_name = f"{state} LGA {i+1}"
        lga_data.append({'State': state, 'LGA': lga_name})

print(f"✅ Created {len(lga_data)} LGAs")

# Step 1: Generate enrollment
enrollment = np.random.normal(loc=5000, scale=2000, size=len(lga_data))
enrollment = np.abs(enrollment).astype(int)
enrollment = np.maximum(enrollment, 500)
enrollment = np.minimum(enrollment, 25000)

# Step 2: Generate teachers
actual_teachers = np.random.normal(loc=150, scale=60, size=len(lga_data))
actual_teachers = np.abs(actual_teachers).astype(int)
actual_teachers = np.maximum(actual_teachers, 20)
actual_teachers = np.minimum(actual_teachers, 800)

# Step 3: Calculate student-to-teacher ratio
students_per_teacher = enrollment / actual_teachers
students_per_teacher = np.round(students_per_teacher, 2)

# Step 4: Calculate base pass rate
base_pass_rate = 50 - (students_per_teacher / 4)

# Step 5: Add noise for actual pass rates
noise = np.random.normal(loc=0, scale=8, size=len(lga_data))
actual_pass_rate = base_pass_rate + noise
actual_pass_rate = np.clip(actual_pass_rate, 0, 100)
actual_pass_rate = np.round(actual_pass_rate, 2)

# Step 6: Generate classrooms
classrooms = np.random.normal(loc=80, scale=30, size=len(lga_data))
classrooms = np.abs(classrooms).astype(int)
classrooms = np.maximum(classrooms, 10)
classrooms = np.minimum(classrooms, 400)

# Step 7: Combine into DataFrame
data = pd.DataFrame({
    'State': [item['State'] for item in lga_data],
    'LGA': [item['LGA'] for item in lga_data],
    'Enrollment': enrollment,
    'Teachers': actual_teachers,
    'Classrooms': classrooms,
    'StudentTeacherRatio': students_per_teacher,
    'BasePassRate': base_pass_rate,
    'ActualPassRate': actual_pass_rate
})

# Step 8: Save to CSV
data.to_csv('education_data.csv', index=False)
print(f"✅ Data saved! Total LGAs: {len(data)}")
print("\nFirst 5 rows:")
print(data.head())