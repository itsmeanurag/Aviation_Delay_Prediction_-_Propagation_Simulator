import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import timedelta

# Add parent dir to path so we can import simulation engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.engine import SimulationEngine

st.set_page_config(layout="wide", page_title="Aviation Delay Simulator")

@st.cache_resource
def load_engine():
    return SimulationEngine(
        xgb_path='models/xgb_base.json',
        lstm_path='models/lstm_prop.keras',
        meta_path='models/meta_model.pkl'
    )

@st.cache_data
def load_data():
    df = pd.read_parquet('data/full_processed.parquet')
    encoders = joblib.load('models/label_encoders.pkl')
    return df, encoders

@st.cache_data
def load_demo_data(_encoders):
    df = pd.read_csv('aviation_industry_demo_dataset.csv')
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])
    
    # Encode origin and destination using existing encoders
    for col in ['origin_iata', 'destination_iata']:
        if col in _encoders:
            try:
                # Fill unknown values with the most common class to avoid transform errors
                known_classes = set(_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else _encoders[col].classes_[0])
                df[col] = _encoders[col].transform(df[col])
            except Exception as e:
                st.warning(f"Warning: Could not encode {col} completely. {e}")

    # Add missing features required by XGBoost (mock values)
    if 'distance_km' not in df.columns:
        df['distance_km'] = 1000.0
    if 'dep_precipitation_mm' not in df.columns:
        df['dep_precipitation_mm'] = 0.0
    if 'dep_storm_indicator' not in df.columns:
        df['dep_storm_indicator'] = 0.0
    if 'dep_flights_this_hour' not in df.columns:
        df['dep_flights_this_hour'] = 10.0
        
    return df

st.title("✈️ Aviation Delay Prediction and Propagation Simulator")

try:
    engine = load_engine()
    df_full, encoders = load_data()
except Exception as e:
    st.error(f"Error loading models or data. Make sure training is complete. {e}")
    st.stop()

# Sidebar for inputs
st.sidebar.header("Data Selection")
dataset_mode = st.sidebar.radio("Choose Dataset", ["Demo Dataset (Upload)", "Full Dataset (Parquet)"])

if dataset_mode == "Demo Dataset (Upload)":
    try:
        df = load_demo_data(encoders)
    except Exception as e:
        st.error(f"Failed to load demo dataset: {e}")
        st.stop()
else:
    df = df_full

st.sidebar.header("Select Simulation Scenario")

tails = df['aircraft_tail'].unique()
if dataset_mode == "Demo Dataset (Upload)":
    # Demo dataset tails are string identifiers (e.g. VT-001)
    tail_options = tails
    tail_map = dict(zip(tails, tails))
else:
    if 'aircraft_tail' in encoders:
        tail_encoder = encoders['aircraft_tail']
        tail_options = tail_encoder.inverse_transform(tails)
        tail_map = dict(zip(tail_options, tails))
    else:
        tail_options = tails
        tail_map = dict(zip(tails, tails))

selected_tail_str = st.sidebar.selectbox("Select Aircraft Tail", tail_options)
selected_tail = tail_map[selected_tail_str]

df_tail = df[df['aircraft_tail'] == selected_tail].sort_values('scheduled_departure')
if df_tail.empty:
    st.warning("No flights available for selected aircraft.")
    st.stop()

# Pick a date based on the aircraft's schedule
dates = df_tail['scheduled_departure'].dt.date.unique()
selected_date = st.sidebar.selectbox("Select Date", dates)

df_tail_day = df_tail[df_tail['scheduled_departure'].dt.date == selected_date].copy()

if df_tail_day.empty:
    st.warning("No flights available on selected date.")
    st.stop()

st.sidebar.subheader("Modify Features (Simulation Override)")
first_flight = df_tail_day.iloc[0]

weather_override = st.sidebar.checkbox("Override Weather for First Flight?")
if weather_override:
    new_wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 150.0, float(first_flight['dep_wind_speed_kmh']))
    # Add safe fallback for precip since it's mocked in demo
    precip_val = float(first_flight.get('dep_precipitation_mm', 0.0))
    new_precip = st.sidebar.slider("Precipitation (mm/hr)", 0.0, 50.0, precip_val)
    new_vis = st.sidebar.slider("Visibility (m)", 0.0, 10000.0, float(first_flight['dep_visibility_m']))
    
    df_tail_day.loc[df_tail_day.index[0], 'dep_wind_speed_kmh'] = new_wind
    df_tail_day.loc[df_tail_day.index[0], 'dep_precipitation_mm'] = new_precip
    df_tail_day.loc[df_tail_day.index[0], 'dep_visibility_m'] = new_vis
    
st.header(f"Flight Chain for Aircraft: {selected_tail_str}")

def decode_airport(code, col_name='origin_iata'):
    if col_name in encoders:
        try:
            return encoders[col_name].inverse_transform([code])[0]
        except ValueError:
            return code
    return code

if st.button("Run Simulation", type="primary"):
    with st.spinner("Running Simulation Engine..."):
        engine.reset_state()
        res_df = engine.process_flights(df_tail_day)
    
    # Display Summary Metrics
    st.subheader("Simulation Summary")
    total_flights = len(res_df)
    total_prop_delay = res_df['cumulative_delay'].iloc[-1]
    max_delay = res_df['final_predicted_delay'].max()
    avg_delay = res_df['final_predicted_delay'].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Flights", total_flights)
    m2.metric("Cumulative Delay", f"{total_prop_delay:.1f} min")
    m3.metric("Max Single Delay", f"{max_delay:.1f} min")
    m4.metric("Avg Delay per Flight", f"{avg_delay:.1f} min")

    st.markdown("---")

    # 1. Aircraft Trail Visualization (Gantt Chart)
    st.subheader("🛫 Aircraft Trail Timeline")
    
    gantt_data = []
    for idx, row in res_df.iterrows():
        orig = decode_airport(row['origin_iata'], 'origin_iata')
        dest = decode_airport(row['destination_iata'], 'destination_iata')
        flight_name = f"Flight {idx+1} ({orig}→{dest})"
        
        # Scheduled
        gantt_data.append(dict(Task=flight_name, Start=row['scheduled_departure'], 
                               Finish=row['scheduled_arrival'], Status='Scheduled'))
        
        # Actual (Delayed)
        actual_dep = row['scheduled_departure'] + timedelta(minutes=row['predicted_base_delay_xgb'])
        # if prop delay exists, it adds to actual dep
        # engine stores predicted_actual_arrival
        actual_arr = row['predicted_actual_arrival']
        # approximate actual dep = actual arr - scheduled flight duration
        flight_dur = row['scheduled_arrival'] - row['scheduled_departure']
        actual_dep = actual_arr - flight_dur

        gantt_data.append(dict(Task=flight_name, Start=actual_dep, 
                               Finish=actual_arr, Status='Predicted Actual'))

    df_gantt = pd.DataFrame(gantt_data)
    
    fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Status",
                            color_discrete_map={"Scheduled": "#1f77b4", "Predicted Actual": "#d62728"},
                            title="Scheduled vs Predicted Flight Trail")
    fig_gantt.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_gantt, use_container_width=True)

    # Display flight details
    with st.expander("View Detailed Flight Log"):
        for idx, row in res_df.iterrows():
            orig = decode_airport(row['origin_iata'], 'origin_iata')
            dest = decode_airport(row['destination_iata'], 'destination_iata')
            
            st.info(f"**Flight {idx+1}: {orig} → {dest}** | Departure: {row['scheduled_departure']}\n\n"
                    f"- Base Delay (XGB): `{row['predicted_base_delay_xgb']:.1f} min`\n"
                    f"- Spill Delay: `{row['calculated_spill_delay']:.1f} min`\n"
                    f"- Propagated Delay (LSTM): `{row['predicted_prop_delay_lstm']:.1f} min`\n"
                    f"- **Final Predicted Delay:** **`{row['final_predicted_delay']:.1f} min`**")

    # 2. Delay Composition
    st.subheader("📊 Delay Breakdown by Component")
    comp_df = res_df[['predicted_base_delay_xgb', 'predicted_prop_delay_lstm', 'calculated_spill_delay']].copy()
    comp_df.index = [f"F{i+1}" for i in range(len(res_df))]
    comp_df.columns = ['Base Delay (XGB)', 'Propagation (LSTM)', 'Spillover']
    
    fig_bar = px.bar(comp_df, barmode='stack', color_discrete_sequence=['#ff7f0e', '#9467bd', '#17becf'])
    fig_bar.update_layout(xaxis_title="Flight in Chain", yaxis_title="Delay (minutes)")
    st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")

    # 3. Explainability (SHAP)
    st.subheader("🧠 SHAP Explainability for Base Delay")
    st.write("Understand which features drove the XGBoost base delay prediction for a specific flight.")
    
    x_features = df_tail_day[engine.xgb_features].copy()
    
    for col in x_features.columns:
        x_features[col] = pd.to_numeric(x_features[col], errors='coerce').fillna(0)
        
    explainer = shap.Explainer(engine.xgb_model.predict, x_features)
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        flight_idx = st.selectbox("Select Flight to Explain", range(1, len(res_df)+1)) - 1
        st.write(f"Explaining Flight {flight_idx+1}")
        
    with col_b:
        x_single = x_features.iloc[[flight_idx]]
        shap_values = explainer(x_single)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap_values[0], show=False)
        st.pyplot(fig2)
