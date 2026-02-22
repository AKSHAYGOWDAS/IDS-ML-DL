import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Dense, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== Load data ==========
df = pd.read_csv("ton_iot.csv")

# ========== Select 8 relevant features ==========
features = [
    "Processor_pct_ Processor_Time",
    "Memory Available Bytes",
    "LogicalDisk(_Total) Disk Reads sec",
    "Network_I(Intel R _82574L_GNC) Bytes Received sec",
    "Memory pct_ Committed Bytes In Use",
    "LogicalDisk(_Total) Avg  Disk Bytes Write",
    "Processor_DPCs_Queued_sec",
    "LogicalDisk(_Total) Avg  Disk sec Read"
]

target = "type"

# ========== Clean data ==========
df_clean = df[features + [target]].replace(r'^\s*$', np.nan, regex=True)
df_clean = df_clean.dropna()

# ========== Encode target labels ==========
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])

# ========== Feature Scaling ==========
scaler = StandardScaler()
X = scaler.fit_transform(df_clean[features])
y = df_clean[target].values

# Save X_train for future scaling (inference)
os.makedirs("saved_models", exist_ok=True)
np.save("saved_models/X_train.npy", X)  # Important for restoring scaler context

# ========== Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# ========== CNN Model ==========
input_layer = Input(shape=(8, 1))
x = Conv1D(64, kernel_size=2, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Conv1D(128, kernel_size=2, activation='relu')(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling1D()(x)
output = Dense(len(np.unique(y)), activation='softmax')(x)

cnn_model = Model(inputs=input_layer, outputs=output)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ========== Train CNN ==========
history = cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# ========== Evaluate CNN ==========
cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
cnn_acc = accuracy_score(y_test, cnn_preds)
print(f"\nCNN Accuracy: {cnn_acc:.4f}")
print(classification_report(y_test, cnn_preds))

# ========== Train XGBoost ==========
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"\nXGBoost Accuracy: {xgb_acc:.4f}")
print(classification_report(y_test, xgb_preds))

# ========== Accuracy Plot ==========
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("CNN Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ========== Save Models ==========
cnn_model.save("saved_models/cnn_intrusion_model.h5")
xgb_model.save_model("saved_models/xgb_intrusion_model.json")

# Save LabelEncoder
with open("saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n✅ Models and encoders saved successfully in 'saved_models/' folder.")
