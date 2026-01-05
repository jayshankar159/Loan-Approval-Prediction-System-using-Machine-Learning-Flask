import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


# =========================
# Load data
# =========================
df = pd.read_csv("train.csv")

X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})


# =========================
# Column types
# =========================
categorical_cols = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed",
    "Property_Area"
]

numerical_cols = [
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term",
    "Credit_History"
]


# =========================
# Pipelines
# =========================
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numerical_pipeline, numerical_cols)
])


# =========================
# Final ML Pipeline
# =========================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Train
# =========================
pipeline.fit(X_train, y_train)


# =========================
# Evaluate
# =========================
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))


# =========================
# Save pipeline
# =========================
joblib.dump(pipeline, "loan_pipeline.pkl")

print("✅ Pipeline trained & saved as loan_pipeline.pkl")
