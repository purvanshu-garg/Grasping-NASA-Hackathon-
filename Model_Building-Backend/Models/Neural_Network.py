# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------------
# 🪐 Example data (replace with your dataset)
# df = pd.read_csv("exoplanet_data.csv")

# Suppose your columns are:
# ['Classification', 'Orbital period', '
#Transit Duration', 'Transit Depth',
#  'Planet radius value', 'Planet Insolation Value',
#  'Planetary Equilibrium Temp.', 'Stellar Effective Temp.',
#  'Stellar Surface Gravity', 'Stellar Radius']
data2=pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\kepler_tess_combined_dummy.csv')
# -----------------------------
# Split features and labels
X = data2.drop(columns=['Classification'], axis=0)
y = data2['Classification']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 🧱 Build Neural Network
def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # binary classification
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(X_train_scaled.shape[1])

# -----------------------------
# ⏱️ Early Stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# -----------------------------
# 🚀 Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 📊 Evaluate
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n🎯 Test Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 📈 (Optional) Plot Training Curves
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()


import joblib

# Save model and scaler
model.save("exoplanet_model.h5")
joblib.dump(scaler, "scaler.pkl")