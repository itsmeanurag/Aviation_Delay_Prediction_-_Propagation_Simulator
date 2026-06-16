import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_lstm_dataset(df):
    # We need to construct X of shape (num_samples, sequence_length, num_features)
    # Timesteps: t-3, t-2, t-1
    # Features: delay, turnaround, congestion
    
    # Drop rows where shifted features are NaN (first few flights per aircraft)
    cols_to_check = []
    for i in [3, 2, 1]:
        cols_to_check.extend([f'delay_t-{i}', f'turnaround_t-{i}', f'congestion_t-{i}'])
    
    df = df.dropna(subset=cols_to_check + ['departure_delay_min'])
    
    X = []
    y = df['departure_delay_min'].values
    
    for i in [3, 2, 1]: # chronological order of timesteps
        timestep_features = df[[f'delay_t-{i}', f'turnaround_t-{i}', f'congestion_t-{i}']].values
        X.append(timestep_features)
        
    X = np.stack(X, axis=1) # shape: (num_samples, 3, 3)
    return X, y

def train_lstm_model(data_path='data/train.parquet', test_path='data/test.parquet'):
    print("Loading data for LSTM...")
    train_df = pd.read_parquet(data_path)
    test_df = pd.read_parquet(test_path)
    
    X_train, y_train = create_lstm_dataset(train_df.iloc[:50000])
    X_test, y_test = create_lstm_dataset(test_df.iloc[:10000])
    
    print(f"Training LSTM on {X_train.shape[0]} sequences...")
    
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(3, 3)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Simple early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("Evaluating LSTM propagation model...")
    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    
    os.makedirs('models', exist_ok=True)
    print("Saving LSTM model...")
    model.save('models/lstm_prop.keras')
    print("LSTM training complete.")

if __name__ == "__main__":
    train_lstm_model()
