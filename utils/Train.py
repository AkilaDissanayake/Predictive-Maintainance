import os
import tensorflow as tf

def train_rul_model(model, X_train, y_train, X_val, y_val, 
                    epochs=50, batch_size=32, loss='mse', 
                    optimizer='adam', model_name="rul_model"):
    
    # Create directory if it doesn't exist
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # CHANGE: Switched extension from .h5 to .keras
    model_path = os.path.join(save_dir, f"{model_name}.keras")
    
    # Compile
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,      # Multiply the LR by 0.2 when triggered (e.g., 0.0001 -> 0.00002)
        patience=5,      # Wait 5 epochs of no improvement before dropping the LR
        min_lr=1e-6,     # Don't let the LR go below this value
        verbose=1        # Print a message when it happens
    )
    # Callbacks: Save best and stop early
    callbacks = [
        reduce_lr,
        # The Checkpoint will now automatically save in the newer format
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
    
    print(f"--- Training {model_name} ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model_path