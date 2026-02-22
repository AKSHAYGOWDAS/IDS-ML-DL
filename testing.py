
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# === SETUP ===
model_path = "veremi_results/veremi_intrusion_model.h5"
dataset_path = "dataset.csv"  # For fitting scaler and label encoder

# === LOAD MODEL ===
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully.")

# === LOAD DATASET FOR SCALER + ENCODER ===
df = pd.read_csv(dataset_path)
features = ['rcvTime', 'pos_0', 'pos_1', 'pos_noise_0', 'pos_noise_1',
            'spd_0', 'spd_1', 'spd_noise_0', 'spd_noise_1',
            'acl_0', 'acl_1', 'acl_noise_0', 'acl_noise_1',
            'hed_0', 'hed_1', 'hed_noise_0', 'hed_noise_1']
X = df[features].values
y = df['attack_type'].values

# Fit scaler and encoder from existing data
scaler = StandardScaler().fit(X)
label_encoder = LabelEncoder().fit(y)
print("✅ Scaler and label encoder initialized.")

# === USER INPUT ===
print("\n🔢 Please enter the following 17 feature values:")
user_input = []
for feature in features:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# Convert to numpy array
user_array = np.array(user_input).reshape(1, -1)

# Scale and reshape
user_scaled = scaler.transform(user_array)
user_reshaped = user_scaled.reshape(1, len(features), 1)

# === PREDICTION ===
predicted_probs = model.predict(user_reshaped)
predicted_class_index = np.argmax(predicted_probs, axis=1)[0]
predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

# === OUTPUT ===
print("\n✅ Prediction complete.")
print(f"Predicted attack type: **{predicted_class_name}**")
