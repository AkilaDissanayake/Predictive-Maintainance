from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_dataframe(df):
    """
    Normalizes every numeric feature in the dataframe to a range of [0, 1].
    Returns the normalized dataframe and the scaler object (useful for inverse scaling later).
    """
    # 1. Initialize the Scaler
    scaler = MinMaxScaler()
    
    # 2. Identify numeric columns (skipping any leftover objects/strings)
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
    
    # 3. Create a copy to avoid modifying the original df in place
    normalized_df = df.copy()
    
    # 4. Fit and Transform the numeric data
    normalized_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return normalized_df, scaler

# Usage:
# df_norm, scaler = normalize_dataframe(df)