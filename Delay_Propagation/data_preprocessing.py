import pandas as pd
import numpy as np
import os
import joblib

def load_and_preprocess(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # 1. Rename columns to match USER INPUT SCHEMA
    rename_mapping = {
        'origin': 'origin_iata',
        'destination': 'destination_iata',
        'sched_dep': 'scheduled_departure',
        'sched_arr': 'scheduled_arrival',
        'orig_temp_c': 'dep_temperature_c',
        'orig_precip_mm_hr': 'dep_precipitation_mm',
        'orig_wind_speed_kmh': 'dep_wind_speed_kmh',
        'orig_wind_dir_deg': 'dep_wind_direction_deg',
        'orig_pressure_hpa': 'dep_pressure_hpa',
        'orig_humidity_pct': 'dep_humidity_pct',
        'orig_storm_indicator': 'dep_storm_indicator',
        'rain_intensity_orig': 'dep_rain_intensity',
        'wind_severity_orig': 'dep_wind_severity',
        'turnaround_min': 'turnaround_time_min',
        'aircraft_id': 'aircraft_tail',
        'dep_delay_min': 'departure_delay_min',
        'flights_last_hour_origin': 'dep_flights_this_hour'
    }
    df = df.rename(columns=rename_mapping)
    
    # Sort chronologically
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])
    df = df.sort_values('scheduled_departure').reset_index(drop=True)
    
    # 2. Derived features
    # distance_km proxy based on duration
    if 'distance_km' not in df.columns:
        if 'flight_duration_min' in df.columns:
            df['distance_km'] = df['flight_duration_min'] * 12.0 # approx 720 km/h
        else:
            df['distance_km'] = (df['scheduled_arrival'] - df['scheduled_departure']).dt.total_seconds() / 60 * 12.0
            
    # Weather and season derived
    df['dep_visibility_m'] = df['orig_visibility_km'] * 1000
    df['dep_is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['dep_is_fog_season'] = df['month'].isin([11, 12, 1, 2]).astype(int)
    df['dep_fog_event'] = (df['dep_visibility_m'] < 1000).astype(int)
    
    # Congestion ratio
    max_flights_per_airport = df.groupby('origin_iata')['dep_flights_this_hour'].transform('max')
    # avoid division by zero
    max_flights_per_airport = np.where(max_flights_per_airport == 0, 1, max_flights_per_airport)
    df['dep_congestion_ratio'] = df['dep_flights_this_hour'] / max_flights_per_airport

    return df

def create_lstm_sequences(df, sequence_length=3):
    print("Creating LSTM sequences...")
    # Sort just in case
    df = df.sort_values(['aircraft_tail', 'scheduled_departure']).reset_index(drop=True)
    
    # Shift features to create sequences
    for i in range(1, sequence_length + 1):
        df[f'delay_t-{i}'] = df.groupby('aircraft_tail')['departure_delay_min'].shift(i)
        df[f'turnaround_t-{i}'] = df.groupby('aircraft_tail')['turnaround_time_min'].shift(i)
        df[f'congestion_t-{i}'] = df.groupby('aircraft_tail')['dep_congestion_ratio'].shift(i)
        
    return df

def main():
    input_file = "india_aviation_flights_2025.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = load_and_preprocess(input_file)
    df = create_lstm_sequences(df, sequence_length=3)
    
    # Drop rows with NaN in shifted features for LSTM training, but keep them for XGBoost if we want to use full data
    # Actually, for XGBoost we don't need the sequence features. We can save full data for XGBoost.
    
    # Drop leakage features completely from the dataset so we don't accidentally use them
    leakage_cols = ['previous_flight_delay_min', 'aircraft_cum_delay_min', 'prev_flight_delay_min']
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])
    
    # Identify categorical columns and encode them for XGBoost
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    cat_cols = ['origin_iata', 'destination_iata', 'aircraft_tail', 'dep_rain_intensity', 'dep_wind_severity']
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string to avoid mixed type errors
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    print("Saving processed data...")
    # Train test split chronologically (e.g., last 20% is test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    train_df.to_parquet('data/train.parquet')
    test_df.to_parquet('data/test.parquet')
    df.to_parquet('data/full_processed.parquet')
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
