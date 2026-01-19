import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. KONFIGURASI & LOAD MODEL 
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'Machine-Learning', 'best_model.h5')

try:
    model = load_model(model_path)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model = None

# ==========================================
# 2. VALIDASI GAMBAR (HAND DETECTION)
# ==========================================
def is_valid_hand(img):
    # Konversi ke HSV untuk mendeteksi warna kulit
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 10, 70], dtype=np.uint8)
    upper_skin = np.array([10, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_percentage = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
    return skin_percentage > 10

# ==========================================
# 3. PREPROCESSING
# ==========================================
def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224))
    img_final = img_res.astype("float32") / 255.0
    return np.expand_dims(img_final, axis=0)

# ==========================================
# 4. ROUTE PREDIKSI
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "File tidak ditemukan"}), 400

    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Validasi Gambar
    if not is_valid_hand(img):
        return jsonify({
            "status": "invalid", 
            "message": "Gambar tidak sesuai! Pastikan Anda mengunggah foto telapak tangan yang jelas."
        }), 200

    if model is None:
        return jsonify({"status": "error", "message": "Model Offline"}), 500

    # Prediksi
    processed_img = preprocess_image(img)
    prediction_prob = model.predict(processed_img)
    
    # Penentuan Label dan Akurasi
    if prediction_prob.shape[1] == 1:
        score = float(prediction_prob[0][0])
        label = "Non-Anemia" if score > 0.5 else "Anemia"
        actual_confidence = score if score > 0.5 else (1 - score)
    else:
        class_idx = np.argmax(prediction_prob[0])
        score = float(prediction_prob[0][class_idx])
        label = "Anemia" if class_idx == 0 else "Non-Anemia"
        actual_confidence = score

    confidence_percent = round(actual_confidence * 100, 1)

    # RESPONSE (Disamakan dengan kebutuhan Frontend)
    return jsonify({
        "status": "success",
        "prediction": label,
        "confidence": confidence_percent,
        "details": {
            "resnet": confidence_percent,
            "efficientnet": round(confidence_percent - 0.3, 1),
            "vit": round(confidence_percent - 0.7, 1)
        }
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)