import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_xgb_model(data_path='data/train.parquet', test_path='data/test.parquet'):
    print("Loading data for XGBoost...")
    train_df = pd.read_parquet(data_path)
    test_df = pd.read_parquet(test_path)
    
    features_xgb = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'month',
        'origin_iata', 'destination_iata', 'distance_km',
        'dep_temperature_c', 'dep_precipitation_mm',
        'dep_wind_speed_kmh', 'dep_visibility_m',
        'dep_humidity_pct', 'dep_storm_indicator',
        'dep_flights_this_hour', 'dep_congestion_ratio',
        'turnaround_time_min'
    ]
    target = 'departure_delay_min'
    
    # Drop rows with NaN in features or target
    train_df = train_df.dropna(subset=features_xgb + [target])
    test_df = test_df.dropna(subset=features_xgb + [target])
    
    X_train = train_df[features_xgb]
    y_train = train_df[target]
    
    X_test = test_df[features_xgb]
    y_test = test_df[target]
    
    # Assert leakage features are not in X_train
    leakage_features = ['previous_flight_delay_min', 'cumulative_aircraft_delay_min']
    for leak in leakage_features:
        assert leak not in X_train.columns, f"Leakage feature {leak} found in XGBoost features!"
    
    print(f"Training XGBoost regressor on {len(X_train)} samples...")
    # Initialize model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10
    )
    
    print("Evaluating XGBoost base model...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    
    os.makedirs('models', exist_ok=True)
    print("Saving XGBoost model...")
    model.save_model('models/xgb_base.json')
    # Also save as pkl for easier usage potentially
    joblib.dump(model, 'models/xgb_base.pkl')
    print("XGBoost training complete.")

if __name__ == "__main__":
    train_xgb_model()
