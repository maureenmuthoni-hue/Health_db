import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Use DATABASE_URL directly from environment
DATABASE_URL = os.getenv("DATABASE_URL")

def get_engine():
    return create_engine(DATABASE_URL)

def load_raw_data():
    df = pd.read_csv("data/healthcare_dataset.csv")
    print(f" Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def store_raw_data(df, engine):
    df.to_sql("raw_healthcare", engine, if_exists="replace", index=False)
    print(" Raw data stored in PostgreSQL (table: raw_healthcare)")

def clean_data(df):
    print("\n Cleaning data...")

    # Drop duplicates
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop irrelevant columns
    drop_cols = [c for c in ["name", "doctor"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert date columns
    for col in ["date_of_admission", "discharge_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Derived feature: length of stay
    if "date_of_admission" in df.columns and "discharge_date" in df.columns:
        df["length_of_stay"] = (df["discharge_date"] - df["date_of_admission"]).dt.days

    # Standardize categorical columns
    cat_cols = ["gender","blood_type","medical_condition","admission_type",
                "insurance_provider","medication","test_results"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols_present = df.select_dtypes(include="object").columns
    for col in cat_cols_present:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop date columns before ML
    df = df.drop(columns=["date_of_admission","discharge_date"], errors="ignore")

    print(f" Cleaned data: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Target distribution:\n{df['test_results'].value_counts()}")
    return df

def store_clean_data(df, engine):
    df.to_sql("cleaned_healthcare", engine, if_exists="replace", index=False)
    print(" Cleaned data stored in PostgreSQL (table: cleaned_healthcare)")

if __name__ == "__main__":
    engine = get_engine()
    df_raw = load_raw_data()
    store_raw_data(df_raw, engine)
    df_clean = clean_data(df_raw)
    store_clean_data(df_clean, engine)
    print("\n Ingestion and cleaning complete!")
