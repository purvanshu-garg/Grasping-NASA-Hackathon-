import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# 1️⃣ Load your datasets
# ==============================
data_1 = pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\kepler_dataset_refined_dummy.csv')
data_2 = pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\TESS_dataset_refined_dummy.csv')

data_1_x = data_1.drop(columns='Classification')
data_2_x = data_2.drop(columns='Classification')

data_1_y = data_1['Classification']
data_2_y = data_2['Classification']

# Train-test split
train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(data_1_x, data_1_y, test_size=0.3, random_state=3)
train_x_2, test_x_2, train_y_2, test_y_2 = train_test_split(data_2_x, data_2_y, test_size=0.3, random_state=3)

train_x_1_2 = pd.concat([train_x_1, train_x_2], ignore_index=True)
train_y_1_2 = pd.concat([train_y_1, train_y_2], ignore_index=True)
test_x_1_2 = pd.concat([test_x_1, test_x_2], ignore_index=True)
test_y_1_2 = pd.concat([test_y_1, test_y_2], ignore_index=True)

# ==============================
# 2️⃣ Define base learners
# ==============================
def build_nn(input_shape, hidden_layers=2, units=64, dropout_rate=0.2):
    model = models.Sequential()
    model.add(layers.Dense(units, activation='relu', input_shape=(input_shape,)))
    for _ in range(hidden_layers - 1):
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==============================
# 3️⃣ Generate out-of-fold predictions
# ==============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_meta_features = np.zeros((train_x_1_2.shape[0], 3))  # RF + NN1 + NN2
test_meta_features = np.zeros((test_x_1_2.shape[0], 3))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x_1_2)):
    print(f"Fold {fold + 1}")

    X_train_fold = train_x_1_2.iloc[train_idx]
    y_train_fold = train_y_1_2.iloc[train_idx]
    X_val_fold = train_x_1_2.iloc[val_idx]
    y_val_fold = train_y_1_2.iloc[val_idx]

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    test_scaled = scaler.transform(test_x_1_2)

    # --- Base learner 1: Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_fold, y_train_fold)
    train_meta_features[val_idx, 0] = rf.predict_proba(X_val_fold)[:, 1]
    test_meta_features[:, 0] += rf.predict_proba(test_x_1_2)[:, 1] / kf.n_splits

    # --- Base learner 2: Neural Network 1 (Kepler) ---
    scaler1 = StandardScaler()
    train_x1_scaled = scaler1.fit_transform(train_x_1)
    val_x1_scaled = scaler1.transform(test_x_1)
    nn1 = build_nn(train_x_1.shape[1], hidden_layers=2)
    nn1.fit(train_x1_scaled, train_y_1, epochs=30, batch_size=32, verbose=0)
    train_meta_features[val_idx, 1] = nn1.predict(X_val_scaled).flatten()
    test_meta_features[:, 1] += nn1.predict(test_scaled).flatten() / kf.n_splits

    # --- Base learner 3: Neural Network 2 (TESS) ---
    scaler2 = StandardScaler()
    train_x2_scaled = scaler2.fit_transform(train_x_2)
    val_x2_scaled = scaler2.transform(test_x_2)
    nn2 = build_nn(train_x_2.shape[1], hidden_layers=2)
    nn2.fit(train_x2_scaled, train_y_2, epochs=30, batch_size=32, verbose=0)
    train_meta_features[val_idx, 2] = nn2.predict(X_val_scaled).flatten()
    test_meta_features[:, 2] += nn2.predict(test_scaled).flatten() / kf.n_splits

# ==============================
# 4️⃣ Train Meta Learner (Neural Network)
# ==============================
meta_scaler = StandardScaler()
train_meta_features_scaled = meta_scaler.fit_transform(train_meta_features)
test_meta_features_scaled = meta_scaler.transform(test_meta_features)

meta_nn = build_nn(train_meta_features_scaled.shape[1], hidden_layers=2, units=32)
meta_nn.fit(train_meta_features_scaled, train_y_1_2, epochs=50, batch_size=32, verbose=0)

# ==============================
# 5️⃣ Evaluate
# ==============================
meta_preds_prob = meta_nn.predict(test_meta_features_scaled)
meta_preds = (meta_preds_prob > 0.5).astype(int)

print("\n--- Meta Learner (Stacked) Results ---")
print("Accuracy:", accuracy_score(test_y_1_2, meta_preds))
print(classification_report(test_y_1_2, meta_preds))

