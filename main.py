from src.data_loader import load_data
from src.feature_engineering import feature_engineering
from src.data_preprocessing import preprocess
from src.feature_selection import feature_selection
from src.clustering import apply_clustering
from src.train_models import train_models
from src.evaluate_models import evaluate_models
from src.threshold_tuning import threshold_tuning
import os
import json
import joblib


if __name__ == "__main__":
    print("🚀 Starting Telecom Churn Prediction Pipeline...\n")

    # =========================
    # Step 1: Load Raw Data
    # =========================
    df = load_data()
    print("📥 Raw data loaded successfully!")
    print(f"📊 Raw data shape: {df.shape}\n")

    # =========================
    # Step 2: Feature Engineering
    # =========================
    df = feature_engineering(df)
    print("🛠️ Feature engineering completed!")
    print(f"📊 Shape after feature engineering: {df.shape}\n")

    # =========================
    # Step 3: Preprocessing
    # =========================
    df = preprocess(df)
    print("🧹 Preprocessing completed!")
    print(f"📊 Shape after preprocessing: {df.shape}\n")

    # =========================
    # Step 4: Feature Selection
    # =========================
    X_cluster_scaled, X_train, X_test, y_train, y_test = feature_selection(df)
    print("🎯 Feature selection completed!")
    print(f"📌 Training shape: {X_train.shape}")
    print(f"📌 Testing shape: {X_test.shape}\n")

    # =========================
    # Step 5: Customer Segmentation (Clustering)
    # =========================
    clustered_df = apply_clustering(df, X_cluster_scaled)
    print("🧩 Customer segmentation completed!")
    print("📌 'customer_segment' column added to dataframe.\n")

    # =========================
    # Step 6: Train Models
    # =========================
    print("🏋️ Training and tuning models...\n")
    tuned_models = train_models(X_train, y_train)

    # =========================
    # Step 7: Evaluate Models
    # =========================
    print("\n📈 Evaluating models...\n")
    results, best_model, best_model_name = evaluate_models(X_test, y_test)

    # =========================
    # Step 8: Threshold Tuning
    # =========================
    print("\n🎯 Performing threshold tuning on best model...\n")
    threshold_df, best_threshold = threshold_tuning(X_test, y_test)

    print(f"\n✅ Final Selected Threshold: {best_threshold}")

    print("\n✅ Pipeline Complete!")
    print("🏁 Telecom Churn Project execution finished successfully.")

    # =========================
    # Step 9: Save Deployment Artifacts
    # =========================

    os.makedirs("artifacts", exist_ok=True)

    # Save final best model
    joblib.dump(best_model, "artifacts/best_xgb_model.pkl")

    # Save final feature columns
    final_feature_columns = X_train.columns.tolist()
    joblib.dump(final_feature_columns, "artifacts/final_feature_columns.pkl")

    # Save threshold config
    config = {
        "threshold": best_threshold
    }

    with open("artifacts/config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\n💾 Deployment artifacts saved successfully!")
    print("📦 Saved files:")
    print("   - artifacts/best_xgb_model.pkl")
    print("   - artifacts/final_feature_columns.pkl")
    print("   - artifacts/config.json")
