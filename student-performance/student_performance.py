import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
data_path = "data/students.csv"
df = pd.read_csv(data_path)

print("\n📘 Dataset Loaded Successfully")
print(df.head())

# ---------------------------------------------------------
# 2. Split Features & Target
# ---------------------------------------------------------
X = df.drop("performance", axis=1)
y = df["performance"]

# ---------------------------------------------------------
# 3. Train – Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. Train Model
# ---------------------------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n🤖 Model training completed.")

# ---------------------------------------------------------
# 5. Evaluate Model
# ---------------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# ---------------------------------------------------------
# 6. Save Model
# ---------------------------------------------------------
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "student_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 Model saved at: {model_path}")

# ---------------------------------------------------------
# 7. Predict for a new student
# ---------------------------------------------------------
new_student = [[6, 88, 78, 1]]  # hours, attendance, prev_score, extra class

pred = model.predict(new_student)[0]
print(f"\n🎯 Predicted Performance for New Student: {pred:.2f}")
