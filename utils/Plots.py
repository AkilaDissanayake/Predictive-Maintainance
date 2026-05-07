import matplotlib.pyplot as plt
import pandas as pd

def plot_multivariate_time_series(df, time_col, feature_cols, up_to=None,title="Feature Trends Over Time"):
    """
    Plots multiple features against a fixed time column in a single combined plot.
    
    Args:
        df (pd.DataFrame): The dataset containing the data.
        time_col (str): The column to be used for the X-axis (e.g., 'timestamp', 'cycle').
        feature_cols (list): A list of strings representing the columns to plot on the Y-axis.
        up_to (int, optional): The number of rows to include in the plot.
        title (str): Custom title for the plot.
    """
    plot_df = df.iloc[:up_to] if up_to is not None else df
    plt.figure(figsize=(12, 6))
    
    # Loop through the list of features and plot each one
    for feature in feature_cols:
        if feature in df.columns:
            plt.plot(plot_df[time_col], plot_df[feature], label=feature, linewidth=1.5)
        else:
            print(f"Warning: Feature '{feature}' not found in DataFrame.")

    # Aesthetics
    plt.title(title, fontsize=14)
    plt.xlabel(time_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel("Measured Values", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_history(history,title="Model Training History"):
    plt.figure(figsize=(12, 4))
    plt.suptitle(title, fontsize=16)
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss (Huber)')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.show()