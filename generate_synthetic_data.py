import numpy as np
import pandas as pd

np.random.seed(42)

# Real Nigerian state names with realistic LGA counts
states = {
    'Abia': 17, 'Adamawa': 21, 'Akwa Ibom': 31, 'Anambra': 21, 'Bauchi': 20,
    'Bayelsa': 8, 'Benue': 23, 'Borno': 27, 'Cross River': 18, 'Delta': 25,
    'Ebonyi': 13, 'Edo': 18, 'Ekiti': 16, 'Enugu': 17, 'FCT': 6,
    'Gombe': 11, 'Imo': 27, 'Jigawa': 27, 'Kaduna': 23, 'Kano': 44,
    'Katsina': 34, 'Kebbi': 21, 'Kogi': 21, 'Kwara': 16, 'Lagos': 20,
    'Nasarawa': 13, 'Niger': 25, 'Ogun': 20, 'Ondo': 18, 'Osun': 30,
    'Oyo': 33, 'Plateau': 17, 'Rivers': 23, 'Sokoto': 23, 'Taraba': 16,
    'Yobe': 17, 'Zamfara': 14
}

rows = []
for state, n_lgas in states.items():
    for i in range(1, n_lgas + 1):
        enrollment = np.random.randint(3000, 60000)
        teachers = np.random.randint(80, 1500)
        classrooms = np.random.randint(30, 800)

        student_teacher_ratio = round(enrollment / teachers, 2)
        pupil_classroom_ratio = round(enrollment / classrooms, 2)

        # Pass rate generated INDEPENDENTLY from resources
        # Only weak correlation — not a formula
        base_pass = 60 - (student_teacher_ratio * 0.4) - (pupil_classroom_ratio * 0.05) + np.random.normal(0, 12)
        base_pass = np.clip(base_pass, 10, 95)
        # Small nudges from resources (realistic but not deterministic)
        if student_teacher_ratio > 60:
            base_pass -= np.random.uniform(2, 8)
        if student_teacher_ratio < 30:
            base_pass += np.random.uniform(1, 5)
        if pupil_classroom_ratio > 80:
            base_pass -= np.random.uniform(1, 5)

        # Inject real-world anomalies
        anomaly_type = np.random.choice(
            ['underperform_severe', 'underperform_mild', 'normal', 'overperform'],
            p=[0.08, 0.18, 0.62, 0.12]
        )
        if anomaly_type == 'underperform_severe':
            base_pass -= np.random.uniform(15, 30)  # Ghost teachers, resource diversion
        elif anomaly_type == 'underperform_mild':
            base_pass -= np.random.uniform(5, 15)
        elif anomaly_type == 'overperform':
            base_pass += np.random.uniform(8, 20)   # High-efficiency star LGA

        actual_pass_rate = round(np.clip(base_pass, 5, 98), 1)

        rows.append({
            'State': state,
            'LGA': f'{state} LGA {i}',
            'Enrollment': enrollment,
            'Teachers': teachers,
            'Classrooms': classrooms,
            'StudentTeacherRatio': student_teacher_ratio,
            'PupilClassroomRatio': pupil_classroom_ratio,
            'ActualPassRate': actual_pass_rate
        })

df = pd.DataFrame(rows)
df.to_csv('education_data.csv', index=False)
print(f"Generated {len(df)} LGAs across {df['State'].nunique()} states")
print(df[['StudentTeacherRatio', 'ActualPassRate']].corr())