from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import math
import urllib.request
import urllib.parse
import json
from datetime import datetime, timezone

# Create Flask application
app = Flask(__name__)

# Locate files relative to base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
features_path = os.path.join(BASE_DIR, "model", "feature_names.pkl")
HISTORY_FILE = os.path.join(BASE_DIR, "predictions_history.json")

# Load trained model and features
model = joblib.load(model_path)
feature_names = joblib.load(features_path)


def get_solar_angles(lat, lon, dt_utc):
    """
    Computes solar zenith and azimuth angles in degrees using pure Python math.
    """
    day_of_year = dt_utc.timetuple().tm_yday
    
    # Fractional year in radians
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (dt_utc.hour - 12.0) / 24.0)
    
    # Equation of time in minutes
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2.0 * gamma) - 0.040849 * math.sin(2.0 * gamma))
    
    # Solar declination angle in radians
    decl = 0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) \
           - 0.006758 * math.cos(2.0 * gamma) + 0.000907 * math.sin(2.0 * gamma) \
           - 0.002697 * math.cos(3.0 * gamma) + 0.00148 * math.sin(3.0 * gamma)
           
    # Time offset in minutes
    time_offset = eqtime + 4.0 * lon
    
    # True solar time in minutes
    t_sol = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0 + time_offset
    
    # Hour angle in degrees
    ha = (t_sol / 4.0) - 180.0
    
    # Convert to radians
    lat_rad = math.radians(lat)
    ha_rad = math.radians(ha)
    
    # Solar Zenith Angle
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    
    # Solar Azimuth Angle (clockwise from North)
    zenith_rad = math.radians(zenith)
    if zenith_rad == 0.0 or math.sin(zenith_rad) == 0.0:
        azimuth = 180.0
    else:
        sin_az = -math.sin(ha_rad) * math.cos(decl)
        cos_az = (math.sin(decl) - math.sin(lat_rad) * cos_zenith) / (math.cos(lat_rad) * math.sin(zenith_rad))
        azimuth = math.degrees(math.atan2(sin_az, cos_az))
        azimuth = (azimuth + 360.0) % 360.0
        
    return zenith, azimuth


def get_angle_of_incidence(zenith, azimuth):
    """
    Computes solar panel Angle of Incidence (AOI) based on a typical 45° tilt facing South.
    """
    z_rad = math.radians(zenith)
    a_rad = math.radians(azimuth)
    
    tilt_rad = math.radians(45.24) # Fitted tilt from dataset
    pan_az_rad = math.radians(180.0) # Fitted panel azimuth (facing South)
    
    cos_aoi = math.cos(z_rad) * math.cos(tilt_rad) + math.sin(z_rad) * math.sin(tilt_rad) * math.cos(a_rad - pan_az_rad)
    cos_aoi = max(-1.0, min(1.0, cos_aoi))
    aoi = math.degrees(math.acos(cos_aoi))
    return aoi


def save_to_history(location, inputs, prediction):
    """
    Saves a prediction record to the local JSON history file.
    """
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "location": location,
        "inputs": inputs,
        "prediction": prediction
    }
    
    history.insert(0, record)
    history = history[:30] # Keep last 30 logs
    
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving prediction history: {e}")


@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)


@app.route("/fetch-weather")
def fetch_weather():
    location = request.args.get("location", "").strip()
    if not location:
        return jsonify({"error": "Location parameter is required"}), 400
    
    try:
        # Geocode the location
        encoded_loc = urllib.parse.quote(location)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_loc}&count=1&language=en&format=json"
        
        req = urllib.request.Request(geo_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            geo_res = json.loads(response.read().decode())
        
        if not geo_res.get("results"):
            return jsonify({"error": f"Location '{location}' not found"}), 404
        
        loc_data = geo_res["results"][0]
        lat = loc_data["latitude"]
        lon = loc_data["longitude"]
        loc_name = loc_data.get("name", "")
        country = loc_data.get("country", "")
        full_loc_name = f"{loc_name}, {country}" if country else loc_name
        
        # Fetch current weather details
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,cloud_cover,wind_gusts_10m,shortwave_radiation,cloud_cover_low"
        
        req_fc = urllib.request.Request(forecast_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_fc) as response:
            fc_res = json.loads(response.read().decode())
        
        current = fc_res.get("current", {})
        if not current:
            return jsonify({"error": "Failed to fetch weather data"}), 500
        
        # Calculate solar angles based on current UTC time
        dt_utc = datetime.now(timezone.utc)
        zenith, azimuth = get_solar_angles(lat, lon, dt_utc)
        aoi = get_angle_of_incidence(zenith, azimuth)
        
        # Map values to the exact feature names expected by the model
        weather_data = {
            "angle_of_incidence": round(aoi, 2),
            "total_cloud_cover_sfc": float(current.get("cloud_cover", 0.0)),
            "zenith": round(zenith, 2),
            "azimuth": round(azimuth, 2),
            "shortwave_radiation_backwards_sfc": float(current.get("shortwave_radiation", 0.0)),
            "total_precipitation_sfc": float(current.get("precipitation", 0.0)),
            "low_cloud_cover_low_cld_lay": float(current.get("cloud_cover_low", 0.0)),
            "wind_gust_10_m_above_gnd": float(current.get("wind_gusts_10m", 0.0)),
            "location_name": full_loc_name
        }
        return jsonify(weather_data)
        
    except Exception as e:
        return jsonify({"error": f"API Error: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []
        location = request.form.get("location_name", "Manual Input").strip()
        if not location:
            location = "Manual Input"

        features_dict = {}
        for feature in feature_names:
            val_str = request.form.get(feature, "0.0")
            value = float(val_str) if val_str else 0.0
            input_data.append(value)
            features_dict[feature] = value

        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        prediction = max(0.0, round(float(prediction[0]), 2)) # Enforce lower bound of 0 kW

        # Save record to history log
        save_to_history(location, features_dict, prediction)

        # Support AJAX updates
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.accept_json:
            return jsonify({
                "prediction": prediction,
                "prediction_text": f"Predicted Solar Power: {prediction} kW",
                "location": location
            })

        return render_template(
            "index.html",
            prediction_text=f"Predicted Solar Power : {prediction} kW",
            feature_names=feature_names
        )
    except Exception as e:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.accept_json:
            return jsonify({"error": str(e)}), 400
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            feature_names=feature_names
        )


@app.route("/history", methods=["GET"])
def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
                return jsonify(history)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])


@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)