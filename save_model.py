import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

print("📂 Loading dataset...")
df = pd.read_csv("student_performance_dataset.csv")

print("🔧 Preprocessing...")
df = pd.get_dummies(df, columns=[
    'Gender',
    'Parental_Education_Level',
    'Internet_Access_at_Home',
    'Extracurricular_Activities'
], drop_first=True)

df['Pass_Fail'] = df['Pass_Fail'].map({'Fail': 0, 'Pass': 1})

# ✅ Drop non-numeric columns (like Student_ID)
df = df.select_dtypes(include=['number', 'bool'])
df = df.astype(int)

X = df.drop(["Pass_Fail", "Final_Exam_Score"], axis=1)
y = df["Pass_Fail"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("✅ Accuracy :", accuracy_score(y_test, y_pred))
print("✅ F1 Score :", f1_score(y_test, y_pred, average='macro'))

joblib.dump(model,           "student_model.pkl")
joblib.dump(scaler,          "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\n🎉 PKL files saved successfully!")
print("📋 Feature columns:", list(X.columns))