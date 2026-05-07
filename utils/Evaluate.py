def evaluate_model(model_path, X_test, y_test):
    """Loads a saved model and evaluates it against multiple metrics."""
    # Load the best version saved during training
    model = tf.keras.models.load_model(model_path)
    
    predictions = model.predict(X_test).flatten()
    
    # Calculate Metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    results = {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2_Score": round(r2, 4)
    }
    
    print("\n--- Evaluation Results ---")
    for metric, value in results.items():
        print(f"{metric}: {value}")
        
    return results, predictions