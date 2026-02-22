import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ========== Load and prepare scaler from training data ==========
df = pd.read_csv("ton_iot.csv")

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

# Clean and preprocess
df_clean = df[features].replace(r'^\s*$', np.nan, regex=True).dropna()
scaler = StandardScaler()
scaler.fit(df_clean[features])  # Fit scaler on full training data

# ========== Load XGBoost model ==========
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("saved_models/xgb_intrusion_model.json")

# ========== Load label encoder ==========
with open("saved_models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== Accept User Input ==========
print("🔐 Enter values for the following 8 system features:")
user_input = []
for feat in features:
    val = float(input(f"{feat}: "))
    user_input.append(val)

# ========== Preprocess and Predict ==========
X_user = np.array(user_input).reshape(1, -1)
X_scaled = scaler.transform(X_user)
pred_class = xgb_model.predict(X_scaled)[0]
pred_label = le.inverse_transform([pred_class])[0]

print(f"\n✅ Predicted Intrusion Type: **{pred_label.upper()}**")
