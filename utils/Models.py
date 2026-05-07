import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def build_lstm_model(input_shape):
    """Deep LSTM for long-term degradation patterns."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Outputting a single RUL value
    ])
    return model

def build_cnn_model(input_shape):
    """1D-CNN for spatial-temporal feature extraction."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(1)
    ])
    return model

def build_mlp_model(input_shape):
    """Simple Multi-Layer Perceptron baseline."""
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model