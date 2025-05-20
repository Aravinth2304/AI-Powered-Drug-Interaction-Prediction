# import os
# import numpy as np
# import pickle
# import tensorflow as tf
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})  # Allow frontend requests from your local server

# # Load API Key from environment variable (SET THIS BEFORE RUNNING)
# API_KEY = os.getenv("API_KEY", "AIzaSyALJ7q6k8kc93NVrMpvWsUaNUJsEaN0OLE")  

# if API_KEY == "AIzaSyALJ7q6k8kc93NVrMpvWsUaNUJsEaN0OLE":
#     print("⚠️ WARNING: Using default API key! Set API_KEY as an environment variable.")

# # Load the trained models
# with open("mlb.pkl", "rb") as f:
#     mlb = pickle.load(f)  # Load MultiLabelBinarizer

# model = tf.keras.models.load_model("drug_interaction_model.h5")  # Load ML model

# def get_feature_vector(drug1, drug2):
#     """Converts medicine names into feature vectors for prediction"""
#     fp1 = np.random.rand(1024)  # Replace with actual embedding function
#     text1 = np.random.rand(768)
#     fp2 = np.random.rand(1024)  
#     text2 = np.random.rand(768)

#     X_input = np.hstack([fp1[:512], text1[:384], fp2[:512], text2[:384]]).reshape(1, -1)
#     return X_input

# @app.route('/predict', methods=['POST'])
# def predict():
#     """API endpoint to predict drug interactions"""
#     data = request.get_json()
#     api_key = request.headers.get("X-API-KEY", "").strip()  

#     if api_key != API_KEY:
#         return jsonify({"error": "Invalid API Key"}), 403

#     if not data or "drug1" not in data or "drug2" not in data:
#         return jsonify({"error": "Both drug names are required"}), 400

#     drug1 = data["drug1"].strip()
#     drug2 = data["drug2"].strip()

#     if not drug1 or not drug2:
#         return jsonify({"error": "Drug names cannot be empty"}), 400

#     try:
#         X_input = get_feature_vector(drug1, drug2)
#         prediction = (model.predict(X_input) > 0.5).astype(int)
#         predicted_interactions = mlb.inverse_transform(prediction)

#         cleaned_interactions = [interaction for interaction in predicted_interactions[0] if len(interaction.split()) < 20]
#         risk_level = "High" if any(word in " ".join(cleaned_interactions).lower() for word in ["fatal", "death", "severe"]) else "Low"

#         result = {
#             "drug1": drug1,
#             "drug2": drug2,
#             "risk_level": risk_level,
#             "side_effects": cleaned_interactions if cleaned_interactions else ["No known interactions"],
#         }
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all frontend requests

# Load API Key from environment variable (SET IT BEFORE RUNNING)
API_KEY = os.getenv("API_KEY", "your_secure_api_key")  

if API_KEY == "your_secure_api_key":
    print("⚠️ WARNING: Using default API key! Set API_KEY as an environment variable.")

# Load the trained models
try:
    with open("mlb_337.pkl", "rb") as f:
        mlb = pickle.load(f)  # Load MultiLabelBinarizer
    model = tf.keras.models.load_model("drug_interaction_model.h5")  # Load ML model
    print("✅ Model and label binarizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    exit(1)

def get_feature_vector(drug1, drug2):
    """Converts drug names into feature vectors for prediction."""
    fp1 = np.random.rand(1024)  # Replace with actual embedding function
    text1 = np.random.rand(768)
    fp2 = np.random.rand(1024)  
    text2 = np.random.rand(768)

    X_input = np.hstack([fp1[:512], text1[:384], fp2[:512], text2[:384]]).reshape(1, -1)
    return X_input

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict drug interactions"""
    data = request.get_json()
    api_key = request.headers.get("X-API-KEY", "").strip()  

    if api_key != API_KEY:
        return jsonify({"error": "Invalid API Key"}), 403

    if not data or "drug1" not in data or "drug2" not in data:
        return jsonify({"error": "Both drug names are required"}), 400

    drug1 = data["drug1"].strip()
    drug2 = data["drug2"].strip()

    if not drug1 or not drug2:
        return jsonify({"error": "Drug names cannot be empty"}), 400

    try:
        print(f"🔹 Received request: {drug1}, {drug2}")

        X_input = get_feature_vector(drug1, drug2)
        print(f"🔹 Feature Vector Shape: {X_input.shape}")

        prediction = (model.predict(X_input) > 0.5).astype(int)
        print(f"🔹 Raw Prediction: {prediction}")

        predicted_interactions = mlb.inverse_transform(prediction)
        print(f"🔹 Predicted Interactions: {predicted_interactions}")
        
        cleaned_interactions = [interaction for interaction in predicted_interactions[0] if len(interaction.split()) < 20]
        risk_level = "High" if any(word in " ".join(cleaned_interactions).lower() for word in ["fatal", "death", "severe"]) else "Low"

        result = {
            "drug1": drug1,
            "drug2": drug2,
            "risk_level": risk_level,
            "side_effects": cleaned_interactions if cleaned_interactions else ["No known interactions"],
        }

        return jsonify(result)

    except Exception as e:
        print(f"❌ Internal Server Error: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
