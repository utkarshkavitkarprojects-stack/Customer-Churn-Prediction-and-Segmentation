import pandas as pd 
import numpy as np


def feature_engineering(df):

    df = df.copy()
   
    df['total_usage'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes'] + df['total_intl_minutes']
    df['total_charges'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge']

    if 'churn' in df.columns:
        df['churn_flag'] = df['churn'].map({'no': 0, 'yes': 1})
        df.drop(columns=['churn'], inplace=True)

    #Avg duration of calls for churned customers
    df['avg_day_call_duration'] = df['total_day_minutes'] / df['total_day_calls']
    df['avg_eve_call_duration'] = df['total_eve_minutes'] / df['total_eve_calls']
    df['avg_night_call_duration'] = df['total_night_minutes'] / df['total_night_calls']
    df['avg_intl_call_duration'] = df['total_intl_minutes'] / df['total_intl_calls']

    df['total_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls']

    df['avg_call_duration'] = df['total_usage']/df['total_calls']

    df['day_usage_share'] = df['total_day_minutes'] / df['total_usage']
    df['eve_usage_share'] = df['total_eve_minutes'] / df['total_usage']
    df['night_usage_share'] = df['total_night_minutes'] / df['total_usage']
    df['intl_usage_share'] = df['total_intl_minutes'] / df['total_usage']

    # Manipulate Features to minimize feature correlation and create new features
    # New engineered features
    df['service_calls_per_month_proxy'] = df['number_customer_service_calls'] / (df['account_length'] + 1)

    df['intl_charge_per_call'] = df['total_intl_charge'] / (df['total_intl_calls'] + 1)
    df['intl_mins_per_call'] = df['total_intl_minutes'] / (df['total_intl_calls'] + 1)

    df['high_service_issue_flag'] = (df['number_customer_service_calls'] >= 4).astype(int)

    df['high_value_customer_flag'] = (df['total_charges'] > df['total_charges'].quantile(0.75)).astype(int)

    df['high_value_dissatisfied_flag'] = (
    (df['total_charges'] > df['total_charges'].quantile(0.75)) &
    (df['number_customer_service_calls'] >= 3)
    ).astype(int)

    df['is_day_heavy_user'] = (df['day_usage_share'] > 0.45).astype(int)
    df['is_intl_heavy_user'] = (df['intl_usage_share'] > df['intl_usage_share'].quantile(0.75)).astype(int)

    df['charge_per_minute'] = df['total_charges'] / (df['total_usage'] + 1)
    df['calls_per_minute'] = df['total_calls'] / (df['total_usage'] + 1)

    # Fill NaNs created due to division
    df.fillna(0, inplace=True)

    #dropping redundant features
    df.drop(columns = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge'], inplace = True)
    # =========================
    # Outlier Treatment
    # =========================
    cap_cols = [
        'service_calls_per_month_proxy',
        'calls_per_minute',
        'avg_call_duration',
        'total_intl_minutes',
        'charge_per_minute'
    ]

    for col in cap_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower=lower, upper=upper)

    # Manual upper-only cap for total_intl_minutes
    df['total_intl_minutes'] = df['total_intl_minutes'].clip(upper=17.25)

    return df