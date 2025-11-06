import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# --- CONFIGURACIÓN ---
DATASET_DIR = "dataset"  # carpeta donde están las imágenes
IMG_SIZE = (96, 96)      # tamaño de entrada
BATCH_SIZE = 16
EPOCHS = 30              # puedes subirlo a 50 si tienes más datos

# --- GENERADORES DE DATOS ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation'
)

# --- MODELO CONVOLUCIONAL SIMPLE ---
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(96,96,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# --- COMPILACIÓN ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- ENTRENAMIENTO ---
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# --- GUARDAR MODELO ---
model.save("modelo_96x96.h5")

print("✅ Modelo guardado como modelo_96x96.h5")
