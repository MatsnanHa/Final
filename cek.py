from flask import Flask, request, jsonify, render_template # type: ignore
import numpy as np
import pickle

# Inisialisasi Flask
app = Flask(__name__)

# Load model yang telah dilatih (file model.pkl harus ada di direktori yang sama)
model = pickle.load(open("final/model/diabetes_model.pkl", "rb"))

# Halaman utama (form input)
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form input pengguna
    data = [
        float(request.form['GenHlth']),
        float(request.form['HighBP']),
        float(request.form['BMI']),
        float(request.form['HighChol']),
        float(request.form['Age']),
        float(request.form['DiffWalk']),
        float(request.form['Income']),
        float(request.form['HeartDiseaseorAttack']),
        float(request.form['PhysHlth'])
    ]

    # Data dalam bentuk numpy array untuk prediksi
    data_array = np.array([data])

    # Prediksi menggunakan model yang sudah diload
    prediction = model.predict(data_array)
    result = 'Diabetes Detected' if prediction[0] == 1 else 'No Diabetes Detected'

    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)
