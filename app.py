from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
app.secret_key = 'abccberfdadeffed'

# === Veremi Model Setup ===
# === Veremi Model Setup ===
veremi_model_path = "results/veremi_intrusion_model.h5"
veremi_dataset_path = "cleaned_balanced_20k_dataset.csv"

veremi_model = tf.keras.models.load_model(
    veremi_model_path,
    compile=False
)


df_veremi = pd.read_csv(veremi_dataset_path)

veremi_features = ['rcvTime', 'pos_0', 'pos_1', 'pos_noise_0', 'pos_noise_1',
                   'spd_0', 'spd_1', 'spd_noise_0', 'spd_noise_1',
                   'acl_0', 'acl_1', 'acl_noise_0', 'acl_noise_1',
                   'hed_0', 'hed_1', 'hed_noise_0', 'hed_noise_1']

# ===== Veremi dataset value suggestions (IoT-like behavior) =====
veremi_suggestions = {
    col: df_veremi[col].dropna().astype(str).unique()[:30]
    for col in veremi_features
}

X_veremi = df_veremi[veremi_features].values
y_veremi = df_veremi['attack_type'].values

veremi_scaler = StandardScaler().fit(X_veremi)
veremi_label_encoder = LabelEncoder().fit(y_veremi)

# === XGBoost Model Setup ===
df_xgb = pd.read_csv("ton_iot.csv")
xgb_features = [
    "Processor_pct_ Processor_Time",
    "Memory Available Bytes",
    "LogicalDisk(_Total) Disk Reads sec",
    "Network_I(Intel R _82574L_GNC) Bytes Received sec",
    "Memory pct_ Committed Bytes In Use",
    "LogicalDisk(_Total) Avg  Disk Bytes Write",
    "Processor_DPCs_Queued_sec",
    "LogicalDisk(_Total) Avg  Disk sec Read"
]
df_clean_xgb = df_xgb[xgb_features].replace(r'^\s*$', np.nan, regex=True).dropna()
xgb_scaler = StandardScaler()
xgb_scaler.fit(df_clean_xgb[xgb_features].values)


xgb_model = xgb.Booster()
xgb_model.load_model("saved_models/xgb_intrusion_model.json")


with open("saved_models/label_encoder.pkl", "rb") as f:
    xgb_label_encoder = pickle.load(f)

# 🔧 FIX MISSING ATTRIBUTES
xgb_model.n_classes_ = len(xgb_label_encoder.classes_)


# === SQLite3 Database Setup ===
def get_db_connection():
    conn = sqlite3.connect('intrusion_ak.db')
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    return conn

# Create users table if it doesn't exist
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

init_db()

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')




@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                           (name, email, password))
            conn.commit()
            cursor.close()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists. Please use a different email.", "danger")

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['email'] = email
            return redirect(url_for('veremi_predict'))
        else:
            error_message = "Invalid credentials, please try again!"

    return render_template('login.html', error_message=error_message)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/veremi_predict', methods=['GET', 'POST'])
def veremi_predict():
    if 'logged_in' not in session:
        flash("Please log in to access the Veremi Prediction page.", "warning")
        return redirect(url_for('login'))

    prediction_result = None
    if request.method == 'POST':
        try:
            user_input = [float(request.form[f]) for f in veremi_features]
            user_array = np.array(user_input).reshape(1, -1)
            user_scaled = veremi_scaler.transform(user_array)
            user_reshaped = user_scaled.reshape(1, len(veremi_features), 1)

            predicted_probs = veremi_model.predict(user_reshaped)
            predicted_index = np.argmax(predicted_probs, axis=1)[0]
            predicted_class = veremi_label_encoder.inverse_transform([predicted_index])[0]

            prediction_result = f"Predicted Attack Type (Veremi Model): {predicted_class}"
        except Exception as e:
            prediction_result = f"Error during prediction: {str(e)}"

    return render_template('veremi_predict.html', features=veremi_features, prediction_result=prediction_result,veremi_suggestions=veremi_suggestions)

@app.route('/xgb_predict', methods=['GET', 'POST'])
def xgb_predict():
    if 'logged_in' not in session:
        flash("Please log in to access the XGBoost Prediction page.", "warning")
        return redirect(url_for('login'))

    prediction_result = None

    if request.method == 'POST':
        try:
            user_input = [float(request.form[f]) for f in xgb_features]

            user_df = pd.DataFrame([user_input], columns=xgb_features)
            scaled_input = xgb_scaler.transform(user_df)

            dmatrix = xgb.DMatrix(scaled_input)
            pred_probs = xgb_model.predict(dmatrix)

            pred_class = int(np.argmax(pred_probs, axis=1)[0])
            pred_label = xgb_label_encoder.inverse_transform([pred_class])[0]

            prediction_result = f"Predicted Intrusion Type (XGBoost): {pred_label}"

        except Exception as e:
            prediction_result = f"Error during XGBoost prediction: {str(e)}"

    return render_template(
        'xgb_predict.html',
        features=xgb_features,
        prediction_result=prediction_result
    )

@app.route('/attack_chart')
def attack_chart():
    graphs = {
        "overall": "graphs/overall_model_performance.png",
        "accuracy": "graphs/training_accuracy.png",
        "loss": "graphs/training_loss.png",
        "cnn_cm": "graphs/cnn_confusion.png",
        "svm_cm": "graphs/svm_confusion.png"
    }
    return render_template('attack_chart.html', graphs=graphs)

@app.route('/chart')
def chart():
    charts = [
        "chart/cnn_performance.png",
        "chart/model_comparison.png",
        "chart/xgboost_performance.png"
    ]
    return render_template("chart.html", charts=charts)

# === Main ===
if __name__ == '__main__':
    print("✅ Flask App Running at http://localhost:5000/")
    app.run(debug=True)
