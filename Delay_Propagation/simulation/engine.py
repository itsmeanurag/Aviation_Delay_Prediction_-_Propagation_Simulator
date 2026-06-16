import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
import joblib
import os

class SimulationEngine:
    def __init__(self, xgb_path='models/xgb_base.json', lstm_path='models/lstm_prop.keras', meta_path='models/meta_model.pkl'):
        print("Initializing Simulation Engine...")
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_path)
        self.lstm_model = tf.keras.models.load_model(lstm_path)
        self.meta_model = joblib.load(meta_path)
        
        self.aircraft_state = {} 
        # aircraft_tail -> { 'last_arrival_time': dt, 'last_delay': float, 'cumulative_delay': float, 'history': [(delay, turnaround, congestion)] }
        
        self.xgb_features = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'month',
            'origin_iata', 'destination_iata', 'distance_km',
            'dep_temperature_c', 'dep_precipitation_mm',
            'dep_wind_speed_kmh', 'dep_visibility_m',
            'dep_humidity_pct', 'dep_storm_indicator',
            'dep_flights_this_hour', 'dep_congestion_ratio',
            'turnaround_time_min'
        ]

    def reset_state(self):
        self.aircraft_state = {}

    def process_flights(self, flights_df):
        # flights_df should be sorted chronologically by scheduled_departure
        flights_df = flights_df.sort_values('scheduled_departure').reset_index(drop=True)
        results = []
        
        for idx, row in flights_df.iterrows():
            tail = row['aircraft_tail']
            
            # Step 1: Retrieve state
            state = self.aircraft_state.get(tail, {
                'last_arrival_time': pd.NaT,
                'last_delay': 0.0,
                'cumulative_delay': 0.0,
                'history': [] # list of (delay, turnaround, congestion)
            })
            
            # Step 2: Compute derived features
            sched_dep = row['scheduled_departure']
            turnaround_time = row['turnaround_time_min']
            
            if pd.isna(state['last_arrival_time']):
                spill_delay = 0.0
            else:
                effective_turnaround = (sched_dep - state['last_arrival_time']).total_seconds() / 60
                if effective_turnaround < turnaround_time:
                    spill_delay = turnaround_time - effective_turnaround
                else:
                    spill_delay = 0.0
                    
            # Step 3: XGBoost prediction
            # Create dataframe to preserve feature names for xgboost
            x_xgb = pd.DataFrame([row[self.xgb_features].to_dict()])
            # Ensure correct types
            for col in x_xgb.columns:
                x_xgb[col] = pd.to_numeric(x_xgb[col], errors='coerce').fillna(0)
                
            delay_xgb = self.xgb_model.predict(x_xgb)[0]
            
            # Step 4: LSTM prediction
            history = state['history']
            if len(history) < 3:
                # If we don't have enough history, pad with zeros
                padded_history = [(0.0, turnaround_time, row['dep_congestion_ratio'])] * (3 - len(history)) + history
            else:
                padded_history = history[-3:]
            
            lstm_input = np.array([padded_history], dtype=np.float32) # shape: (1, 3, 3)
            delay_lstm = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            
            # Step 5: FINAL DELAY via Meta Model
            x_meta = np.array([[delay_xgb, delay_lstm, spill_delay]])
            final_delay = self.meta_model.predict(x_meta)[0]
            
            # Ensure delay is not wildly negative
            final_delay = max(0.0, final_delay)
            
            # Step 6: Update state
            actual_arrival = row['scheduled_arrival'] + pd.Timedelta(minutes=final_delay)
            
            # append to history
            new_history = history + [(final_delay, turnaround_time, row['dep_congestion_ratio'])]
            
            self.aircraft_state[tail] = {
                'last_arrival_time': actual_arrival,
                'last_delay': final_delay,
                'cumulative_delay': state['cumulative_delay'] + final_delay,
                'history': new_history
            }
            
            result = row.to_dict()
            result['predicted_base_delay_xgb'] = float(delay_xgb)
            result['predicted_prop_delay_lstm'] = float(delay_lstm)
            result['calculated_spill_delay'] = float(spill_delay)
            result['final_predicted_delay'] = float(final_delay)
            result['predicted_actual_arrival'] = actual_arrival
            result['cumulative_delay'] = self.aircraft_state[tail]['cumulative_delay']
            
            results.append(result)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test engine logic if directly executed
    if os.path.exists('models/xgb_base.json') and os.path.exists('models/lstm_prop.keras') and os.path.exists('models/meta_model.pkl'):
        engine = SimulationEngine()
        print("Engine initialized successfully.")
    else:
        print("Models not found. Please train models first.")
