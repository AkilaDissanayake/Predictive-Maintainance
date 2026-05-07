import numpy as np

def create_sequences(X_data, y_data, window_size=10):
    """
    Converts 2D data into 3D sequences for LSTM.
    X_data: Normalized features (DataFrame or np.array)
    y_data: Normalized target (DataFrame or np.array)
    window_size: How many previous steps to look at
    """
    X_seq, y_seq = [], []
    
    # Convert to numpy if they are DataFrames
    X_values = X_data.values if hasattr(X_data, 'values') else X_data
    y_values = y_data.values if hasattr(y_data, 'values') else y_data
    
    for i in range(len(X_values) - window_size):
        X_seq.append(X_values[i : (i + window_size)])
        y_seq.append(y_values[i + window_size])
        
    return np.array(X_seq), np.array(y_seq)