from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Cargar el modelo, el scaler y el encoder
# Se cargan una sola vez al iniciar la API 
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    # Recibir datos enviados por el usuario
    data = request.json

    # Convertir a array en el MISMO ORDEN del entrenamiento
    features = np.array([[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]])

    # Escalar igual que en el entrenamiento
    scaled_features = scaler.transform(features)

    # Hacer predicción con el modelo cargado
    prediction = model.predict(scaled_features)[0]

    # Convertir número → nombre de clase
    class_name = encoder.inverse_transform([prediction])[0]

    # Respuesta con ambos valores
    return jsonify({
        "prediccion_numero": int(prediction),   # 0, 1, 2
        "prediccion_clase": class_name          # Nombre real
    })


@app.route("/", methods=["GET"])
def home():
    return "API de Clasificación Iris funcionando correctamente."

if __name__ == "__main__":
    # Ejecutar la API
    app.run(host="0.0.0.0", port=5000, debug=True)
