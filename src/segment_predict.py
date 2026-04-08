# src/segment_predict.py

import os
import json
import joblib
import pandas as pd

from src.feature_engineering import feature_engineering
from src.data_preprocessing import preprocess


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

KMEANS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "cluster_scaler.pkl")
CLUSTER_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "cluster_feature_columns.pkl")
SEGMENT_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "segment_mapping.json")


# =========================================================
# LOAD CLUSTERING ARTIFACTS
# =========================================================
def load_clustering_artifacts():
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    cluster_scaler = joblib.load(SCALER_PATH)
    cluster_feature_columns = joblib.load(CLUSTER_FEATURES_PATH)

    with open(SEGMENT_MAPPING_PATH, "r") as f:
        segment_mapping = json.load(f)

    segment_mapping = {int(k): v for k, v in segment_mapping.items()}

    return kmeans_model, cluster_scaler, cluster_feature_columns, segment_mapping


# =========================================================
# PREPARE CLUSTER INPUT
# =========================================================
def prepare_cluster_input(input_df, cluster_feature_columns):
    """
    Apply feature engineering + preprocessing and prepare cluster input.
    """
    df = input_df.copy()

    df = feature_engineering(df)
    df = preprocess(df)

    # Add missing cluster columns if needed
    for col in cluster_feature_columns:
        if col not in df.columns:
            df[col] = 0

    X_cluster = df[cluster_feature_columns].copy()
    return X_cluster


# =========================================================
# PREDICT SEGMENT
# =========================================================
def predict_segment(input_df):
    """
    Predict customer segment for one or multiple customers.
    Returns dataframe with cluster_id and customer_segment only.
    """
    kmeans_model, cluster_scaler, cluster_feature_columns, segment_mapping = load_clustering_artifacts()

    X_cluster = prepare_cluster_input(input_df, cluster_feature_columns)
    X_cluster_scaled = cluster_scaler.transform(X_cluster)

    cluster_ids = kmeans_model.predict(X_cluster_scaled)
    segment_labels = [segment_mapping.get(int(cid), f"Cluster {cid}") for cid in cluster_ids]

    result_df = pd.DataFrame({
        "cluster_id": cluster_ids,
        "customer_segment": segment_labels
    })

    return result_df