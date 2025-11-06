import os
import tensorflow as tf

model_path = "keras_model.h5"
assert os.path.exists(model_path), "No encuentro keras_model.h5 en la carpeta actual"

model = tf.keras.models.load_model(model_path)
model.summary()

print("Forma de entrada:", model.input_shape)
print("Número de clases:", model.output_shape[-1])

size_mb = os.path.getsize(model_path) / 1024 / 1024
print(f"Tamaño del modelo: {size_mb:.2f} MB")

