# ✈️ Aviation Delay Prediction & Propagation Simulator

## Overview
The Aviation Delay Prediction & Propagation Simulator is a machine learning project designed to predict flight departure delays and simulate how delays propagate across aircraft schedules. The system combines XGBoost, LSTM neural networks, and a stacking meta-model to improve prediction accuracy while modeling real-world operational impacts.

## Features
- Flight delay prediction using XGBoost
- Delay propagation modeling using LSTM
- Ensemble prediction with a meta-model
- Aircraft state tracking and simulation engine
- Data preprocessing and feature engineering pipeline
- Interactive Streamlit dashboard
- Delay analysis and visualization tools

- ## Model Architecture

### 1. XGBoost Model
Predicts baseline departure delays using:
- Time-based features
- Route information
- Weather conditions
- Airport congestion metrics
- Aircraft turnaround times

### 2. LSTM Model
Learns delay propagation patterns from previous flights of the same aircraft using:
- Historical delays
- Turnaround times
- Congestion ratios

### 3. Meta Model
Combines:
- XGBoost predictions
- LSTM predictions
- Spill delay calculations

to generate the final delay prediction.

## Installation

```bash
git clone <repository-url>
