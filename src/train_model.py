import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                              recall_score, f1_score, confusion_matrix,
                              classification_report)
from xgboost import XGBClassifier
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'healthdb')
DB_USER = os.getenv('DB_USER', 'healthuser')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'healthpass123')
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_engine():
    return create_engine(DB_URL)

def load_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM cleaned_healthcare", engine)
    print(f"✅ Loaded {len(df)} rows from database")
    return df

def preprocess(df):
    df = df.copy()

    # Drop columns not needed for prediction
    df = df.drop(columns=["hospital", "room_number"], errors="ignore")

    # Encode target
    target_encoder = LabelEncoder()
    df["test_results_encoded"] = target_encoder.fit_transform(df["test_results"])
    print(f"  Classes: {list(target_encoder.classes_)}")

    # Features and target
    X = df.drop(columns=["test_results", "test_results_encoded"])
    y = df["test_results_encoded"]

    # Encode all remaining categorical columns
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, target_encoder, encoders

def evaluate_model(name, model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"📊 Model: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=classes, zero_division=0)}")
    return accuracy_score(y_test, y_pred)

def train():
    df = load_data()
    X, y, target_encoder, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ Train size: {len(X_train)}, Test size: {len(X_test)}")

    classes = list(target_encoder.classes_)
    results = {}

    # --- Model 1: Logistic Regression ---
    print("\n🔧 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results["LogisticRegression"] = (lr, evaluate_model("Logistic Regression", lr, X_test, y_test, classes))

    # --- Model 2: Random Forest ---
    print("\n🔧 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["RandomForest"] = (rf, evaluate_model("Random Forest", rf, X_test, y_test, classes))

    # --- Model 3: XGBoost ---
    print("\n🔧 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, random_state=42,
                         eval_metric="mlogloss", verbosity=0)
    xgb.fit(X_train, y_train)
    results["XGBoost"] = (xgb, evaluate_model("XGBoost", xgb, X_test, y_test, classes))

    # --- Pick best model ---
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\n🏆 Best model: {best_name} (Accuracy: {results[best_name][1]:.4f})")

    # --- Save best model + encoders ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(target_encoder, "models/target_encoder.pkl")
    joblib.dump(encoders, "models/feature_encoders.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")
    print("✅ Saved: models/best_model.pkl")
    print("✅ Saved: models/target_encoder.pkl")
    print("✅ Saved: models/feature_encoders.pkl")
    print("✅ Saved: models/feature_columns.pkl")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    train()
