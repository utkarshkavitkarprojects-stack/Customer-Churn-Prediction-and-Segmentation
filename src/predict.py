# src/predict.py

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import feature_engineering
from src.data_preprocessing import preprocess
from src.segment_predict import predict_segment


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_xgb_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "final_feature_columns.pkl")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")


# =========================================================
# LOAD CHURN ARTIFACTS
# =========================================================
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    final_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        threshold = config.get("threshold", 0.60)
    else:
        threshold = 0.60

    return model, final_feature_columns, threshold


# =========================================================
# ALIGN FEATURES
# =========================================================
def align_features(df, final_feature_columns):
    """
    Ensure processed dataframe matches training feature columns.
    """
    for col in final_feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[final_feature_columns]
    return df


# =========================================================
# BUSINESS OUTPUT HELPERS
# =========================================================
def get_risk_level(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    elif prob < 0.80:
        return "High Risk"
    else:
        return "Very High Risk"


def get_retention_action(prob):
    if prob < 0.30:
        return "No immediate action required."
    elif prob < 0.60:
        return "Monitor customer and consider targeted offers."
    elif prob < 0.80:
        return "Proactive retention campaign recommended."
    else:
        return "Immediate personalized retention intervention required."

# =========================================================
# SEGMENT STRATEGY FUNCTION
# =========================================================
def get_segment_strategy(segment_name):
    """
    Return segment-based business strategy.
    """
    strategy_map = {
        "High-Value Heavy Usage Customers":
            "Prioritize retention with loyalty offers, premium plans, and proactive service quality monitoring.",

        "Routine Low-Engagement Customers":
            "Maintain engagement with personalized offers, simple plan upgrades, and periodic value-based outreach.",

        "International-Focused Budget Customers":
            "Offer cost-effective international bundles and targeted pricing plans to improve retention and value perception.",

        "Service-Sensitive Support-Heavy Customers":
            "Focus on service recovery, faster issue resolution, and customer support improvements to reduce dissatisfaction."
    }

    return strategy_map.get(segment_name, "No segment strategy available.")


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================
def predict_customer(input_df):
    """
    Predict churn + segment for one or multiple customers.

    Parameters
    ----------
    input_df : pd.DataFrame
        Raw customer input dataframe

    Returns
    -------
    pd.DataFrame
        Final prediction output
    """
    # Load model artifacts
    model, final_feature_columns, threshold = load_artifacts()

    # Keep raw copy
    raw_df = input_df.copy()

    # -----------------------------
    # Churn Prediction Pipeline
    # -----------------------------
    df = feature_engineering(raw_df.copy())
    df = preprocess(df)

    X_final = align_features(df, final_feature_columns)

    churn_proba = model.predict_proba(X_final)[:, 1]
    churn_pred = (churn_proba >= threshold).astype(int)

    # -----------------------------
    # Business Output
    # -----------------------------
    result_df = raw_df.copy()
    result_df["churn_probability"] = np.round(churn_proba, 4)
    result_df["churn_prediction"] = churn_pred
    result_df["risk_level"] = result_df["churn_probability"].apply(get_risk_level)
    result_df["retention_action"] = result_df["churn_probability"].apply(get_retention_action)

    # -----------------------------
    # Segment Prediction
    # -----------------------------
    segment_df = predict_segment(raw_df.copy())

    result_df["cluster_id"] = segment_df["cluster_id"].values
    result_df["customer_segment"] = segment_df["customer_segment"].values

    result_df["segment_strategy"] = result_df["customer_segment"].apply(get_segment_strategy)

    return result_df


# =========================================================
# SINGLE CUSTOMER WRAPPER
# =========================================================
def predict_single_customer(customer_dict):
    input_df = pd.DataFrame([customer_dict])
    return predict_customer(input_df)


# =========================================================
# TEST RUN
# =========================================================
if __name__ == "__main__":

    sample_customer = {
        "account_length": 120,
        "international_plan": "no",
        "voice_mail_plan": "yes",
        "voice_mail_messages": 25,

        "total_day_minutes": 250.5,
        "total_day_calls": 110,
        "total_day_charge": 42.59,

        "total_eve_minutes": 180.2,
        "total_eve_calls": 95,
        "total_eve_charge": 15.32,

        "total_night_minutes": 220.4,
        "total_night_calls": 100,
        "total_night_charge": 9.92,

        "total_intl_minutes": 12.5,
        "total_intl_calls": 4,
        "total_intl_charge": 3.38,

        "number_customer_service_calls": 2
    }

    result = predict_single_customer(sample_customer)

    print("\n🔮 Final Prediction Result:")
    print(result[[
        "churn_probability",
        "churn_prediction",
        "risk_level",
        "cluster_id",
        "customer_segment",
        "retention_action",
        "segment_strategy"
    ]])