import pandas as pd
import numpy as np

def process_timestamps(df, col_name='timestamp'):
    """
    Converts a timestamp string column into numerical features:
    1. Seconds_Elapsed: Total seconds from the start of the dataset.
    2. Hour_Sin/Cos: Cyclical representation of the time of day.
    """
    # 1. Convert to datetime objects
    df[col_name] = pd.to_datetime(df[col_name])
    
    # 2. Calculate Seconds Elapsed (Great for tracking degradation)
    start_time = df[col_name].min()
    df['seconds_elapsed'] = (df[col_name] - start_time).dt.total_seconds()
    
    # 3. Cyclical Hour Encoding (Helps model understand 11PM is near 12AM)
    hour = df[col_name].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # 4. Drop the original string/object column so the model doesn't crash
    df = df.drop(columns=[col_name])
    
    return df

# Example Usage:
# df = pd.read_csv("your_data.csv")
# df = process_timestamps(df, col_name='time')