# ============================================================
# VeReMi IDS - SHELL INPUT TEST SCRIPT
# You ENTER values in terminal
# OUTPUT: ONLY EXACT ATTACK TYPE
# ============================================================

import joblib
import pandas as pd
import os

MODEL_DIR = "trained_models"

FEATURES = [
    'rcvTime',
    'pos_0', 'pos_1', 'pos_noise_0', 'pos_noise_1',
    'spd_0', 'spd_1', 'spd_noise_0', 'spd_noise_1',
    'acl_0', 'acl_1', 'acl_noise_0', 'acl_noise_1',
    'hed_0', 'hed_1', 'hed_noise_0', 'hed_noise_1'
]

# ============================================================
# READ INPUTS FROM SHELL
# ============================================================
print("\nEnter values for prediction:\n")

user_values = {}
for feature in FEATURES:
    user_values[feature] = float(input(f"{feature}: "))

X = pd.DataFrame([user_values])[FEATURES]

# ============================================================
# STAGE-1: FAMILY PREDICTION
# ============================================================
stage1_model = joblib.load(f"{MODEL_DIR}/stage1_family_model.pkl")
stage1_scaler = joblib.load(f"{MODEL_DIR}/stage1_family_scaler.pkl")
stage1_encoder = joblib.load(f"{MODEL_DIR}/stage1_family_encoder.pkl")

X_scaled = stage1_scaler.transform(X)
family_idx = stage1_model.predict(X_scaled)
family = stage1_encoder.inverse_transform(family_idx)[0]

# ============================================================
# STAGE-2: ATTACK TYPE PREDICTION
# ============================================================
stage2_model_path = f"{MODEL_DIR}/stage2_{family}_model.pkl"

if os.path.exists(stage2_model_path):
    stage2_model = joblib.load(stage2_model_path)
    stage2_scaler = joblib.load(f"{MODEL_DIR}/stage2_{family}_scaler.pkl")
    stage2_encoder = joblib.load(f"{MODEL_DIR}/stage2_{family}_encoder.pkl")

    X2 = stage2_scaler.transform(X)
    attack_idx = stage2_model.predict(X2)
    attack_type = stage2_encoder.inverse_transform(attack_idx)[0]
else:
    attack_type = family

# ============================================================
# FINAL OUTPUT (ONLY ONE LINE)
# ============================================================
print("\nPredicted Attack Type:")
print(attack_type)
