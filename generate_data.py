import pandas as pd
import numpy as np

print("📊 Generating structured student performance dataset...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for 500 students
n_students = 500
study_hours = np.random.uniform(2, 20, n_students)       # 2 to 20 hours a week
attendance = np.random.uniform(60, 100, n_students)      # 60% to 100% attendance
assignments = np.random.uniform(50, 100, n_students)     # 50% to 100% completion scores

# Calculate a final grade based on a mathematical formula + some random noise
final_grade = (study_hours * 1.5) + (attendance * 0.4) + (assignments * 0.3) + np.random.normal(0, 3, n_students)
final_grade = np.clip(final_grade, 0, 100) # Keep grades between 0 and 100

# Create a Pandas DataFrame
df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Assignments_Completed': assignments,
    'Final_Grade': final_grade
})

# Save to a clean CSV file
df.to_csv('students.csv', index=False)
print("✅ 'students.csv' successfully created with 500 records!")
