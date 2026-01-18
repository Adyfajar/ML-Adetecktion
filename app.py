from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app) # Mengizinkan akses dari frontend HTML

def is_palm_detected(img):
    """
    Logika deteksi apakah gambar telapak tangan.
    Gunakan model deteksi tangan (seperti MediaPipe) atau analisis warna kulit.
    """
    # Contoh sederhana: Cek proporsi warna kulit (H-S-V)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    skin_percentage = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
    return skin_percentage > 30 # Jika warna kulit > 30%, dianggap tangan

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # 1. Validasi Telapak Tangan
    if not is_palm_detected(img):
        return jsonify({"status": "invalid"}), 200

    # 2. Jalankan Model ML Anda di sini (Xception / MobileNet / ResNet)
    # prediction_result = model.predict(img)
    
    # CONTOH DATA DINAMIS (Ganti dengan hasil dataset/model Anda)
    results = {
        "status": "success",
        "prediction": "Non-Anemic", # Atau "Anemic"
        "confidence": 0,
        "details": {
            "resnet": 96.1,
            "efficientnet": 87.7,
            "vit": 55.1
        }
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    