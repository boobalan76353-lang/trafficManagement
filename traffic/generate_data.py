import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Settings
num_rows = 1000
locations = ['Downtown', 'Suburbs', 'Highway', 'Industrial Zone']
weather_conditions = ['Clear', 'Rainy', 'Cloudy', 'Foggy']

# Generate data
np.random.seed(42)
data = {
    'DateTime': [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(num_rows)],
    'Location': np.random.choice(locations, num_rows),
    'Weather': np.random.choice(weather_conditions, num_rows),
}

df = pd.DataFrame(data)

# Extract features from DateTime
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# Base vehicle count influenced by hour and location
def calculate_vehicle_count(row):
    # Rush hours: 8-10 AM, 5-7 PM
    base = 100
    if 8 <= row['Hour'] <= 10 or 17 <= row['Hour'] <= 19:
        base = 500
    
    if row['Location'] == 'Downtown':
        base += 200
    elif row['Location'] == 'Highway':
        base += 300
    
    # Add noise
    return int(base + np.random.normal(0, 50))

df['Vehicle_Count'] = df.apply(calculate_vehicle_count, axis=1)

# Congestion Level mapping
def get_congestion(row):
    vc = row['Vehicle_Count']
    if row['Weather'] in ['Rainy', 'Foggy']:
        vc *= 1.3
    
    if vc < 300:
        return 'Low'
    elif vc < 600:
        return 'Medium'
    else:
        return 'High'

df['Congestion_Level'] = df.apply(get_congestion, axis=1)

# Save to CSV
df.to_csv('data/traffic_data.csv', index=False)
print("Dataset created successfully at data/traffic_data.csv")
