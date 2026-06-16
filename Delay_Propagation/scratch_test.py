import pandas as pd
import joblib

df = pd.read_csv('aviation_industry_demo_dataset.csv')
encoders = joblib.load('models/label_encoders.pkl')

print("Encoders available:", encoders.keys())

for col in ['origin_iata', 'destination_iata']:
    if col in encoders:
        try:
            df[col] = encoders[col].transform(df[col])
            print(f"Successfully encoded {col}")
        except Exception as e:
            print(f"Error encoding {col}: {e}")

if 'aircraft_tail' in encoders:
    try:
        df['aircraft_tail_encoded'] = encoders['aircraft_tail'].transform(df['aircraft_tail'])
        print(f"Successfully encoded aircraft_tail")
    except Exception as e:
        print(f"Error encoding aircraft_tail: {e}")
