import tensorflow as tf

# Cargar tu modelo original
modelo = tf.keras.models.load_model("modelo_96x96.h5")

# Crear el convertidor a TensorFlow Lite
convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)

# Activar la optimización
convertidor.optimizations = [tf.lite.Optimize.DEFAULT]

# Cuantización completa a int8 (para ESP32)
def represent_data():
    for _ in range(100):
        # valores aleatorios con forma 224x224x3 entre 0 y 1
       yield [tf.random.uniform([1, 96, 96, 3], 0, 1)]

convertidor.representative_dataset = represent_data
convertidor.target_spec.supported_types = [tf.int8]
convertidor.inference_input_type = tf.uint8
convertidor.inference_output_type = tf.uint8

modelo_tflite = convertidor.convert()

# Guardar el modelo reducido
with open("model_quantized.tflite", "wb") as f:
    f.write(modelo_tflite)

print("✅ Modelo cuantizado guardado como model_quantized.tflite")
