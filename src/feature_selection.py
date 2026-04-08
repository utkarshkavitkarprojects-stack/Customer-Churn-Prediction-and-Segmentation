# Selecting features for clustering as per RFM analysis 
# # Select your features wisely to avoid overfitting
# cluster_features = [
#     # Value
#     'total_charges',

#     # Usage intensity
#     'total_usage',
#     'total_calls',
#     'avg_call_duration',

#     # Usage pattern
#     'day_usage_share',
#     'eve_usage_share',
#     'night_usage_share',
#     'intl_usage_share',

#     # Friction / dissatisfaction
#     'number_customer_service_calls',
#     'service_calls_per_month_proxy',

#     # International behavior
#     'total_intl_minutes',
#     'intl_mins_per_call'
# ]

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def feature_selection(df):
    df = df.copy()
    cluster_features = [
    'total_charges',
    'total_usage',
    'total_calls',
    'avg_call_duration',
    'day_usage_share',
    'eve_usage_share',
    'night_usage_share',
    'intl_usage_share',
    'number_customer_service_calls',
    'service_calls_per_month_proxy',
    'total_intl_minutes',
    'intl_mins_per_call'
    ]

    X_cluster = df[cluster_features].copy()

    #Applying feature scaling for clustering
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    df = df.copy()
   
    #Dropping redundant average duration columns
    df.drop(['avg_day_call_duration', 'avg_eve_call_duration', 'avg_night_call_duration', 'avg_intl_call_duration'], axis=1, inplace=True)

    #Feature selection using random forest 
    X_fs = df.drop('churn_flag', axis=1)
    y_fs = df['churn_flag']

    rf_fs = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
    )

    rf_fs.fit(X_fs, y_fs)

    feature_importance = pd.DataFrame({
    'feature': X_fs.columns,
    'importance': rf_fs.feature_importances_
    }).sort_values(by='importance', ascending=False)

    top_20_features = feature_importance.head(20)['feature'].tolist()
    df = df[top_20_features].copy()

    #We are going to drop some columns after redundancy/correlation pruning 

    redundant_after_top20 = [
    'calls_per_minute',        # mirror of avg_call_duration
    'intl_charge_per_call',    # duplicate of intl_mins_per_call
    'charge_per_minute',       # overlaps with usage/value
    'high_value_customer_flag' # compressed threshold version
    ]

    df = df.drop(columns=redundant_after_top20, errors='ignore')

    X = df.copy()
    y = y_fs

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

    return X_cluster_scaled, X_train, X_test,y_train, y_test

    



