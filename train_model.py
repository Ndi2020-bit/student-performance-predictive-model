import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("🤖 Loading dataset and initializing Machine Learning Pipeline...")

# 1. Load the data using Pandas
df = pd.read_csv('students.csv')

# 2. Split data into Features (X) and Target (y)
X = df[['Study_Hours', 'Attendance', 'Assignments_Completed']] # What the model looks at
y = df['Final_Grade'] # What the model is trying to predict

# 3. Train/Test Split (80% training data, 20% evaluation data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

# 4. Initialize and Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("🎯 Model training complete.")

# 5. Evaluate Performance on Unseen Test Data
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n🏆 --- MODEL PERFORMANCE METRICS ---")
print(f"Mean Absolute Error: {mae:.2f} percentage points")
print(f"R² Variance Score (Accuracy): {r2 * 100:.2f}%")

# 6. Make a live inference prediction for a custom student
custom_student = [[15.0, 92.0, 85.0]] # 15 hours study, 92% attendance, 85% assignments
predicted_score = model.predict(custom_student)
print(f"\n🔮 Prediction for custom student: {predicted_score[0]:.2f}%")
