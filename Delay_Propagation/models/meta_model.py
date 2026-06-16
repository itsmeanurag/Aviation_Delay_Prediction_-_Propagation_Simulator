import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from lstm_prop import create_lstm_dataset

def compute_spill_delay(df):
    df = df.copy()
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    # For training the meta model, we need the actual arrival of the previous flight.
    # In the CSV, we originally had 'actual_arr', but we might have dropped it or it's named 'actual_arrival'
    # Wait, in data_preprocessing, we didn't drop 'actual_arr' explicitly. We renamed sched_arr to scheduled_arrival.
    # We should have 'actual_arr' in the dataset.
    if 'actual_arr' in df.columns:
        df['actual_arr'] = pd.to_datetime(df['actual_arr'])
        df['last_arrival_time'] = df.groupby('aircraft_tail')['actual_arr'].shift(1)
    else:
        # Approximate last arrival time using scheduled_arrival + arr_delay_min (or departure_delay_min)
        # Assuming we just use departure_delay_min of previous flight for simplicity if actual_arr is missing
        df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])
        # shift departure delay
        prev_delay = df.groupby('aircraft_tail')['departure_delay_min'].shift(1).fillna(0)
        df['last_arrival_time'] = df.groupby('aircraft_tail')['scheduled_arrival'].shift(1) + pd.to_timedelta(prev_delay, unit='m')
        
    df['effective_turnaround'] = (df['scheduled_departure'] - df['last_arrival_time']).dt.total_seconds() / 60
    df['spill_delay'] = np.maximum(0, df['turnaround_time_min'] - df['effective_turnaround'])
    df['spill_delay'] = df['spill_delay'].fillna(0)
    return df['spill_delay'].values

def train_meta_model(data_path='data/train.parquet', test_path='data/test.parquet'):
    print("Loading data for Meta Model...")
    train_df = pd.read_parquet(data_path).iloc[:10000]
    test_df = pd.read_parquet(test_path).iloc[:2000]
    
    # We must drop rows that cannot be used by LSTM (due to sequence shift NaNs)
    cols_to_check = []
    for i in [3, 2, 1]:
        cols_to_check.extend([f'delay_t-{i}', f'turnaround_t-{i}', f'congestion_t-{i}'])
    
    train_df = train_df.dropna(subset=cols_to_check + ['departure_delay_min'])
    test_df = test_df.dropna(subset=cols_to_check + ['departure_delay_min'])
    
    # Get Spill Delay
    spill_train = compute_spill_delay(train_df)
    spill_test = compute_spill_delay(test_df)
    
    # Get XGBoost predictions
    print("Generating XGBoost predictions...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('models/xgb_base.json')
    
    features_xgb = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'month',
        'origin_iata', 'destination_iata', 'distance_km',
        'dep_temperature_c', 'dep_precipitation_mm',
        'dep_wind_speed_kmh', 'dep_visibility_m',
        'dep_humidity_pct', 'dep_storm_indicator',
        'dep_flights_this_hour', 'dep_congestion_ratio',
        'turnaround_time_min'
    ]
    xgb_pred_train = xgb_model.predict(train_df[features_xgb])
    xgb_pred_test = xgb_model.predict(test_df[features_xgb])
    
    # Get LSTM predictions
    print("Generating LSTM predictions...")
    lstm_model = tf.keras.models.load_model('models/lstm_prop.keras')
    X_lstm_train, _ = create_lstm_dataset(train_df)
    X_lstm_test, _ = create_lstm_dataset(test_df)
    
    lstm_pred_train = lstm_model.predict(X_lstm_train, verbose=0).flatten()
    lstm_pred_test = lstm_model.predict(X_lstm_test, verbose=0).flatten()
    
    # Construct Meta Model Features
    X_meta_train = np.column_stack((xgb_pred_train, lstm_pred_train, spill_train))
    y_meta_train = train_df['departure_delay_min'].values
    
    X_meta_test = np.column_stack((xgb_pred_test, lstm_pred_test, spill_test))
    y_meta_test = test_df['departure_delay_min'].values
    
    print("Training Ridge Meta Model...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_meta_train)
    
    print("Evaluating Meta Model...")
    y_pred_test = meta_model.predict(X_meta_test)
    rmse = np.sqrt(mean_squared_error(y_meta_test, y_pred_test))
    mae = mean_absolute_error(y_meta_test, y_pred_test)
    r2 = r2_score(y_meta_test, y_pred_test)
    
    print(f"Meta Model RMSE: {rmse:.2f}")
    print(f"Meta Model MAE: {mae:.2f}")
    print(f"Meta Model R2: {r2:.4f}")
    print(f"Learned Weights (XGB, LSTM, Spill): {meta_model.coef_}")
    print(f"Intercept: {meta_model.intercept_}")
    
    joblib.dump(meta_model, 'models/meta_model.pkl')
    print("Meta Model training complete.")

if __name__ == "__main__":
    train_meta_model()
