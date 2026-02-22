# ============================================================
# VeReMi Intrusion Detection System - FINAL TRAINING SCRIPT
# 1) 5-Class High Accuracy Model
# 2) Hierarchical 2-Stage Model
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

import xgboost as xgb

# ============================================================
# PATHS
# ============================================================
DATASET_PATH = "cleaned_balanced_20k_dataset.csv"
SAVE_DIR = "trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATASET_PATH)

FEATURES = [
    'rcvTime',
    'pos_0', 'pos_1', 'pos_noise_0', 'pos_noise_1',
    'spd_0', 'spd_1', 'spd_noise_0', 'spd_noise_1',
    'acl_0', 'acl_1', 'acl_noise_0', 'acl_noise_1',
    'hed_0', 'hed_1', 'hed_noise_0', 'hed_noise_1'
]

# ============================================================
# ATTACK FAMILY MAPPING
# ============================================================
family_map = {
    "ConstPos": "Position",
    "RandomPos": "Position",
    "ConstPosOffset": "Position",
    "RandomPosOffset": "Position",

    "ConstSpeed": "Speed",
    "RandomSpeed": "Speed",
    "ConstSpeedOffset": "Speed",
    "RandomSpeedOffset": "Speed",

    "DoS": "DoS",
    "DoSDisruptive": "DoS",
    "DoSRandom": "DoS",
    "DoSDisruptiveSybil": "DoS",
    "DoSRandomSybil": "DoS",

    "DataReplay": "Replay",
    "DataReplaySybil": "Replay",

    "Normal": "Normal"
}

# ============================================================
# MAP + CLEAN (CRITICAL FIX)
# ============================================================
df["family"] = df["attack_type"].map(family_map)

# DROP UNMAPPED ATTACKS (avoids NaN / float error)
df = df.dropna(subset=["family"])

# FORCE STRING TYPE (avoids classification_report crash)
df["family"] = df["family"].astype(str)
df["attack_type"] = df["attack_type"].astype(str)

# ============================================================
# PART 1 — 5-CLASS MODEL
# ============================================================
print("\n========== TRAINING 5-CLASS MODEL ==========")

X = df[FEATURES]
y = df["family"]

le_5 = LabelEncoder()
y = le_5.fit_transform(y)

scaler_5 = StandardScaler()
X = scaler_5.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    stratify=y,
    test_size=0.25,
    random_state=42
)

model_5 = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softmax",
    num_class=len(le_5.classes_),
    eval_metric="mlogloss",
    tree_method="hist"
)

model_5.fit(X_tr, y_tr)
y_pred = model_5.predict(X_te)

print("\n5-CLASS CLASSIFICATION REPORT\n")
print(classification_report(
    y_te,
    y_pred,
    target_names=[str(c) for c in le_5.classes_]
))

joblib.dump(model_5, f"{SAVE_DIR}/veremi_5class_model.pkl")
joblib.dump(scaler_5, f"{SAVE_DIR}/veremi_5class_scaler.pkl")
joblib.dump(le_5, f"{SAVE_DIR}/veremi_5class_encoder.pkl")

# ============================================================
# PART 2 — HIERARCHICAL MODEL
# ============================================================

# ----------------------------
# STAGE 1 — FAMILY CLASSIFIER
# ----------------------------
print("\n========== TRAINING STAGE-1 FAMILY MODEL ==========")

X = df[FEATURES]
y = df["family"]

le_1 = LabelEncoder()
y = le_1.fit_transform(y)

scaler_1 = StandardScaler()
X = scaler_1.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    stratify=y,
    test_size=0.25,
    random_state=42
)

stage1 = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softmax",
    num_class=len(le_1.classes_),
    eval_metric="mlogloss",
    tree_method="hist"
)

stage1.fit(X_tr, y_tr)
y_pred = stage1.predict(X_te)

print("\nSTAGE-1 FAMILY CLASSIFICATION REPORT\n")
print(classification_report(
    y_te,
    y_pred,
    target_names=[str(c) for c in le_1.classes_]
))

joblib.dump(stage1, f"{SAVE_DIR}/stage1_family_model.pkl")
joblib.dump(scaler_1, f"{SAVE_DIR}/stage1_family_scaler.pkl")
joblib.dump(le_1, f"{SAVE_DIR}/stage1_family_encoder.pkl")

# ----------------------------
# STAGE 2 — SUB-ATTACK MODELS
# ----------------------------
print("\n========== TRAINING STAGE-2 SUB-ATTACK MODELS ==========")

for fam in df["family"].unique():

    subset = df[df["family"] == fam]

    # Skip if only one sub-attack
    if subset["attack_type"].nunique() < 2:
        continue

    print(f"\n--- Training Stage-2 Model for FAMILY: {fam} ---")

    X_f = subset[FEATURES]
    y_f = subset["attack_type"]

    le_2 = LabelEncoder()
    y_f = le_2.fit_transform(y_f)

    scaler_2 = StandardScaler()
    X_f = scaler_2.fit_transform(X_f)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_f, y_f,
        stratify=y_f,
        test_size=0.25,
        random_state=42
    )

    model_2 = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=len(le_2.classes_),
        eval_metric="mlogloss",
        tree_method="hist"
    )

    model_2.fit(X_tr, y_tr)
    y_pred = model_2.predict(X_te)

    print(classification_report(
        y_te,
        y_pred,
        target_names=[str(c) for c in le_2.classes_]
    ))

    joblib.dump(model_2, f"{SAVE_DIR}/stage2_{fam}_model.pkl")
    joblib.dump(scaler_2, f"{SAVE_DIR}/stage2_{fam}_scaler.pkl")
    joblib.dump(le_2, f"{SAVE_DIR}/stage2_{fam}_encoder.pkl")

print("\n✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
