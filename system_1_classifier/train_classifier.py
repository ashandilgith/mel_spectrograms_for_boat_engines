import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

SPEC_PATH = "system_1_classifier/data/spectrograms"
MODEL_DIR = "system_1_classifier/models"
from audio_to_mel import CLASSES

def load_data():
    X, y = [], []
    for class_idx, class_name in enumerate(CLASSES):
        for f in glob.glob(os.path.join(SPEC_PATH, class_name, "*.npy")):
            X.append(np.load(f))
            y.append(class_idx)
    return np.array(X)[..., np.newaxis], np.array(y)

def build_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(128, 44, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_cnn(len(CLASSES))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16, callbacks=callbacks)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "fault_classifier.keras"))
    with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
        json.dump(CLASSES, f)