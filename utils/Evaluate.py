import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model_path, X_test, y_test):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Get predictions
    predictions = model.predict(X_test).flatten()
    
    # --- USE SKLEARN METRICS ---
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    results = {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2_Score": round(float(r2), 4)
    }
    
    print("\n--- Evaluation Results ---")
    for metric, value in results.items():
        print(f"{metric}: {value}")
        
    return results, predictions