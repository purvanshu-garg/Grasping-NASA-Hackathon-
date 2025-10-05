import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# ============   ==================
# 1️⃣ Load your datasets and Train (Run once, then save models/scalers)
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

def train_and_save_models():
    # Load data
    data_1 = pd.read_csv('kepler_dataset_refined_dummy.csv')
    data_2 = pd.read_csv('TESS_dataset_refined_dummy.csv')

    data_1_x = data_1.drop(columns='Classification')
    data_2_x = data_2.drop(columns='Classification')

    data_1_y = data_1['Classification']
    data_2_y = data_2['Classification']

    train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(data_1_x, data_1_y, test_size=0.3, random_state=3)
    train_x_2, test_x_2, train_y_2, test_y_2 = train_test_split(data_2_x, data_2_y, test_size=0.3, random_state=3)

    train_x_1_2 = pd.concat([train_x_1, train_x_2], ignore_index=True)
    train_y_1_2 = pd.concat([train_y_1, train_y_2], ignore_index=True)
    test_x_1_2 = pd.concat([test_x_1, test_x_2], ignore_index=True)
    test_y_1_2 = pd.concat([test_y_1, test_y_2], ignore_index=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_meta_features = np.zeros((train_x_1_2.shape[0], 3))
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
        nn1 = build_nn(train_x_1.shape[1], hidden_layers=2)
        nn1.fit(train_x1_scaled, train_y_1, epochs=30, batch_size=32, verbose=0)
        train_meta_features[val_idx, 1] = nn1.predict(scaler1.transform(X_val_fold)).flatten()
        test_meta_features[:, 1] += nn1.predict(scaler1.transform(test_x_1_2)).flatten() / kf.n_splits

        # --- Base learner 3: Neural Network 2 (TESS) ---
        scaler2 = StandardScaler()
        train_x2_scaled = scaler2.fit_transform(train_x_2)
        nn2 = build_nn(train_x_2.shape[1], hidden_layers=2)
        nn2.fit(train_x2_scaled, train_y_2, epochs=30, batch_size=32, verbose=0)
        train_meta_features[val_idx, 2] = nn2.predict(scaler2.transform(X_val_fold)).flatten()
        test_meta_features[:, 2] += nn2.predict(scaler2.transform(test_x_1_2)).flatten() / kf.n_splits

    # Meta-learner
    meta_scaler = StandardScaler()
    train_meta_features_scaled = meta_scaler.fit_transform(train_meta_features)

    meta_nn = build_nn(train_meta_features_scaled.shape[1], hidden_layers=2, units=32)
    meta_nn.fit(train_meta_features_scaled, train_y_1_2, epochs=50, batch_size=32, verbose=0)

    # Save everything needed for prediction
    joblib.dump(rf, 'rf_model.joblib')
    joblib.dump(scaler, 'rf_scaler.joblib')
    joblib.dump(scaler1, 'nn1_scaler.joblib')
    nn1.save('nn1_model.h5')
    joblib.dump(scaler2, 'nn2_scaler.joblib')
    nn2.save('nn2_model.h5')
    joblib.dump(meta_scaler, 'meta_scaler.joblib')
    meta_nn.save('meta_nn_model.h5')

# Uncomment next line to run training and save models
train_and_save_models()

# ==============================
# 2️⃣ Predict on User Input CSV
# ==============================
def predict_user_csv(user_csv_path):
    # Load trained models and scalers
    rf = joblib.load('rf_model.joblib')
    rf_scaler = joblib.load('rf_scaler.joblib')
    nn1_scaler = joblib.load('nn1_scaler.joblib')
    nn2_scaler = joblib.load('nn2_scaler.joblib')
    meta_scaler = joblib.load('meta_scaler.joblib')
    nn1 = tf.keras.models.load_model('nn1_model.h5')
    nn2 = tf.keras.models.load_model('nn2_model.h5')
    meta_nn = tf.keras.models.load_model('meta_nn_model.h5')

    # Load user data
    user_data = pd.read_csv(user_csv_path)
    if 'Classification' in user_data.columns:
        user_data_x = user_data.drop(columns='Classification')
    else:
        user_data_x = user_data

    # Each base model expects the same features as training
    # Get meta features for user input
    # 1. Random Forest
    rf_features = rf_scaler.transform(user_data_x)
    rf_pred = rf.predict_proba(user_data_x)[:, 1]

    # 2. NN1 (Kepler)
    nn1_features = nn1_scaler.transform(user_data_x)
    nn1_pred = nn1.predict(nn1_features).flatten()

    # 3. NN2 (TESS)
    nn2_features = nn2_scaler.transform(user_data_x)
    nn2_pred = nn2.predict(nn2_features).flatten()

    # Stack meta-features
    user_meta_features = np.vstack([rf_pred, nn1_pred, nn2_pred]).T

    # Scale meta-features
    user_meta_scaled = meta_scaler.transform(user_meta_features)

    # Meta learner prediction
    meta_pred_prob = meta_nn.predict(user_meta_scaled)
    meta_pred = (meta_pred_prob > 0.5).astype(int)

    # Output prediction
    user_data['Predicted_Classification'] = meta_pred
    print(user_data[['Predicted_Classification']])
    return user_data

# Example usage:
result = predict_user_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\TESS_dataset_refined_dummy.csv')
data=pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\TESS_dataset_refined_dummy.csv')
y=data['Classification']
print(accuracy_score(y,result))
print(classification_report(y,result))