import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv('data/traffic_data.csv')

# Feature Engineering
# Convert DateTime to useful features if not already done in generator
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# Encoding categorical variables
le_location = LabelEncoder()
df['Location_Enc'] = le_location.fit_transform(df['Location'])

le_weather = LabelEncoder()
df['Weather_Enc'] = le_weather.fit_transform(df['Weather'])

le_congestion = LabelEncoder()
df['Congestion_Enc'] = le_congestion.fit_transform(df['Congestion_Level'])

# Features and Target
X = df[['Hour', 'DayOfWeek', 'Location_Enc', 'Weather_Enc', 'Vehicle_Count']]
y = df['Congestion_Enc']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save models and encoders
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/traffic_model.pkl')
joblib.dump(le_location, 'models/le_location.pkl')
joblib.dump(le_weather, 'models/le_weather.pkl')
joblib.dump(le_congestion, 'models/le_congestion.pkl')

print("Models and encoders saved successfully in models/ folder.")
