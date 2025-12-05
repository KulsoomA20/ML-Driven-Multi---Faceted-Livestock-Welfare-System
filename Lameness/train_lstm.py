# src/train_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def build_lstm_model(timesteps, n_features, n_classes=3):
    model = Sequential([
        Masking(mask_value=0., input_shape=(timesteps, n_features)),
        LSTM(256),  
        Dropout(0.3),  
        Dense(128, activation='relu'),  
        Dense(n_classes, activation='softmax'),
    ])
    optimizer = Adam(learning_rate=0.0005)  
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Load features ===
base_dir = r"C:\Users\kulso\OneDrive\Desktop\Project\DLC_PROJECT\Classificationnn"
features_path = f"{base_dir}\\features.csv"

df = pd.read_csv(features_path)

# Extract features and labels
X = df.drop(columns=["label", "video"]).values
y = df["label"].values

# Encode string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape to (samples, timesteps, features_per_step)
X = X.reshape((X.shape[0], 1, X.shape[1]))  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train model
timesteps = X_train.shape[1]
n_features = X_train.shape[2]
n_classes = len(np.unique(y))

model = build_lstm_model(timesteps, n_features, n_classes)
model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=70,  
    batch_size=16,  
    verbose=1
)

# Save model
model.save(f"{base_dir}\\lstm_model_tuned1.h5")
print(f" Model saved to {base_dir}\\lstm_model_tuned1.h5")
