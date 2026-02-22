import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout,
    Dense, MultiHeadAttention, GlobalAveragePooling1D
)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

# ================================
# TensorFlow precision
# ================================
tf.keras.backend.set_floatx('float32')

# ================================
# LOCAL PATHS (CHANGE IF NEEDED)
# ================================
dataset_path = "cleaned_balanced_20k_dataset.csv"
save_dir = "resultss"

os.makedirs(save_dir, exist_ok=True)

# ================================
# LOAD DATASET
# ================================
df = pd.read_csv(dataset_path)
df = df.iloc[:20000, :]
print("Dataset shape:", df.shape)

# ================================
# FEATURES & LABEL
# ================================
features = [
    'rcvTime', 'pos_0', 'pos_1', 'pos_noise_0', 'pos_noise_1',
    'spd_0', 'spd_1', 'spd_noise_0', 'spd_noise_1',
    'acl_0', 'acl_1', 'acl_noise_0', 'acl_noise_1',
    'hed_0', 'hed_1', 'hed_noise_0', 'hed_noise_1'
]

X = df[features].values
y = df['attack_type'].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42
)

# ================================
# CNN + ATTENTION MODEL
# ================================
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    residual = Conv1D(256, 1, padding='same')(inputs)
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005, epsilon=1e-5, clipnorm=1.0
)

model = build_model((X_reshaped.shape[1], 1), num_classes)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# TRAIN MODEL
# ⚠️ Reduce epochs if system is slow
# ================================
history = model.fit(
    X_train, y_train,
    epochs=4000,
    batch_size=256,
    validation_data=(X_test, y_test)
)

# ================================
# PLOTS
# ================================
plt.plot(history.history['accuracy'])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(save_dir, "accuracy.png"))
plt.show()

plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(save_dir, "loss.png"))
plt.show()

# ================================
# EVALUATION
# ================================
y_pred = np.argmax(model.predict(X_train), axis=1)
report = classification_report(y_train, y_pred, target_names=label_encoder.classes_)
conf_matrix = confusion_matrix(y_train, y_pred)

print(report)
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("CNN + Attention Confusion Matrix")
plt.savefig(os.path.join(save_dir, "cnn_confusion.png"))
plt.show()

# Save report
with open(os.path.join(save_dir, "cnn_report.txt"), "w") as f:
    f.write(report)
    f.write("\n\n")
    f.write(str(conf_matrix))

model.save(os.path.join(save_dir, "veremi_intrusion_model.h5"))

# ================================
# SVM MODEL
# ================================
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42
)

svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train_svm, y_train_svm)

svm_pred = svm.predict(X_train_svm)
svm_report = classification_report(y_train_svm, svm_pred, target_names=label_encoder.classes_)
svm_cm = confusion_matrix(y_train_svm, svm_pred)

print(svm_report)
print(svm_cm)

sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("SVM Confusion Matrix")
plt.savefig(os.path.join(save_dir, "svm_confusion.png"))
plt.show()

with open(os.path.join(save_dir, "svm_report.txt"), "w") as f:
    f.write(svm_report)
    f.write("\n\n")
    f.write(str(svm_cm))
