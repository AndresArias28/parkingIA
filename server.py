from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Carga tu modelo (ajusta la ruta y nombre)
model = tf.keras.models.load_model('modelo_96x96.h5')

# Define tus clases
CLASSES = ['carro', 'moto', 'vacio']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = image.resize((96, 96))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    result = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

    return jsonify(result)

if __name__ == '__main__':
    # Permite conexiones externas cambiando host a '0.0.0.0'
    app.run(host='0.0.0.0', port=5000, debug=True)
