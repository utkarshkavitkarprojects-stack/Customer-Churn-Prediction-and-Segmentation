import numpy as np 
import pandas as pd 

def preprocess(df):

    df = df.copy()
   
    print(f"✅ Loaded data with shape: {df.shape}")

    # 🧹 Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"🧹 Dropped rows with missing values: {missing_count}")
    else:
        print("✅ No missing values found")

    # 🧹 Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"🧹 Dropped duplicate rows: {duplicate_count}")
    else:
        print("✅ No duplicates found")

    # 🔠 Encode categorical columns (if any)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"🔠 Encoded categorical columns: {categorical_cols}")
    else:
        print("✅ No categorical columns to encode")
    
    return df 