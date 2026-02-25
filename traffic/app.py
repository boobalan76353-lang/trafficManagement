from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import qrcode
import socket

app = Flask(__name__)

def get_local_ip():
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# Load models and encoders
MODEL_PATH = 'models/traffic_model.pkl'
LE_LOCATION_PATH = 'models/le_location.pkl'
LE_WEATHER_PATH = 'models/le_weather.pkl'
LE_CONGESTION_PATH = 'models/le_congestion.pkl'

model = joblib.load(MODEL_PATH)
le_location = joblib.load(LE_LOCATION_PATH)
le_weather = joblib.load(LE_WEATHER_PATH)
le_congestion = joblib.load(LE_CONGESTION_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    # Use the specified IP address
    ip = "192.168.1.100"
    url = f"http://{ip}:5000"
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    qr_path = os.path.join(app.root_path, 'static', 'qr_code.png')
    img.save(qr_path)
    
    return render_template('about.html', qr_code='qr_code.png', ip_address=url)

@app.route('/prediction')
def prediction():
    locations = le_location.classes_.tolist()
    weathers = le_weather.classes_.tolist()
    return render_template('prediction.html', locations=locations, weathers=weathers)

@app.route('/dataset')
def dataset():
    df = pd.read_csv('data/traffic_data.csv').head(20)
    data = df.to_dict(orient='records')
    return render_template('dataset.html', data=data)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        dt_str = data['datetime']
        location = data['location']
        weather = data['weather']
        volume = int(data['volume'])

        dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M')
        hour = dt.hour
        day_of_week = dt.weekday()

        loc_enc = le_location.transform([location])[0]
        weather_enc = le_weather.transform([weather])[0]

        features = np.array([[hour, day_of_week, loc_enc, weather_enc, volume]])
        prediction_enc = model.predict(features)[0]
        prediction_label = le_congestion.inverse_transform([prediction_enc])[0]
        
        # Confidence score (using predict_proba)
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100

        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
