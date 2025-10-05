import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# ==============================
# 1️⃣ Load data
# ==============================
data_1 = pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\kepler_dataset_refined_dummy.csv')
data_2 = pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\TESS_dataset_refined_dummy.csv')


# ==============================
# 2️⃣ Feature engineering
# ==============================
def feature_engineering(df):
    df = df.copy()
    
    # 1. New ratio features
    df['TransitDepth_to_Radius'] = df['Transit Depth'] / (df['Planet radius value'] + 1e-5)
    df['Insolation_to_EquilibriumTemp'] = df['Planet Insolation Value'] / (df['Planetary Equilibrium Temp.'] + 1e-5)
    df['Radius_to_StellarRadius'] = df['Planet radius value'] / (df['Stellar Radius'] + 1e-5)

    # 2. Log-transformed features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in ['Orbital period', 'Transit Duration', 'Transit Depth', 
                'Planet radius value', 'Planet Insolation Value']:
        if col in numerical_cols.tolist():
             df[f'log_{col}'] = np.log1p(df[col])

    # Select *only* the columns that will be used for polynomial features
    poly_cols_to_use = [
        'Orbital period', 'Transit Duration', 'Transit Depth', 
        'Planet radius value', 'Stellar effective temperature',
        'Stellar Surface Gravity', 'Stellar Radius', 
        'Stellar log(g)', 'Stellar Mass', 'Planetary Equilibrium Temp.',
        'Planet Insolation Value'
    ]
    
    # Filter for columns that actually exist in the DataFrame
    poly_input_df = df[[col for col in poly_cols_to_use if col in df.columns]]

    # 3. Interaction features (The Fix is here!)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(poly_input_df)
    poly_feature_names = poly.get_feature_names_out(poly_input_df.columns)
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

    # Identify and drop the original features from the poly output
    # This leaves *only* the interaction terms (e.g., 'A B', but not 'A' or 'B')
    original_feature_names = poly_input_df.columns.tolist()
    interaction_only_cols = [col for col in df_poly.columns if all(name in col for name in original_feature_names) and (' ' in col)]
    
    df_interactions = df_poly[interaction_only_cols].reset_index(drop=True)
    df = df.reset_index(drop=True)

    # 4. Concatenate only the new interaction terms
    df = pd.concat([df, df_interactions], axis=1)
    
    # Final check: remove any duplicate columns that may have arisen from prior steps
    df = df.loc[:,~df.columns.duplicated()].copy() 

    return df

data_1_fe = feature_engineering(data_1)
data_2_fe = feature_engineering(data_2)

# ==============================
# 3️⃣ Feature selection
# ==============================
X_1 = data_1_fe.drop(columns='Classification')
y_1 = data_1_fe['Classification']

selector = SelectKBest(score_func=f_classif, k=30)
X_1_selected = selector.fit_transform(X_1, y_1)
selected_features = X_1.columns[selector.get_support()]

X_2 = data_2_fe[selected_features]
y_2 = data_2_fe['Classification']

# ==============================
# 4️⃣ Data balancing with SMOTE
# ==============================
sm = SMOTE(random_state=42)
X_res_1, y_res_1 = sm.fit_resample(X_1_selected, y_1)
X_res_2, y_res_2 = sm.fit_resample(X_2, y_2)

# ==============================
# 5️⃣ Combine datasets for training
# ==============================
X_combined = np.vstack((X_res_1, X_res_2))
y_combined = np.hstack((y_res_1, y_res_2))

X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

# ==============================
# 6️⃣ Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 7️⃣ Base models
# ==============================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

# Neural Network as base learner
def build_nn(input_shape):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = build_nn(X_train_scaled.shape[1])
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
nn_train_preds = nn_model.predict(X_train_scaled).flatten()
nn_test_preds = nn_model.predict(X_test_scaled).flatten()

# ==============================
# 8️⃣ Stacking ensemble
# ==============================
stack = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb_clf)
    ],
    final_estimator=LogisticRegression()
)

stack.fit(X_train_scaled, y_train)
stack_preds = stack.predict(X_test_scaled)

# ==============================
# 9️⃣ Meta learner Neural Network
# ==============================
meta_X_train = np.column_stack([stack.predict_proba(X_train_scaled)[:,1], nn_train_preds])
meta_X_test = np.column_stack([stack.predict_proba(X_test_scaled)[:,1], nn_test_preds])

meta_nn = build_nn(meta_X_train.shape[1])
meta_nn.fit(meta_X_train, y_train, epochs=50, batch_size=32, verbose=0)

final_preds = (meta_nn.predict(meta_X_test) > 0.5).astype(int)

# ==============================
# 10️⃣ Evaluation
# ==============================
print("Stacking Ensemble Accuracy:", accuracy_score(y_test, stack_preds))
print("Meta Learner Accuracy:", accuracy_score(y_test, final_preds))
print("\nClassification Report:\n", classification_report(y_test, final_preds))
print(confusion_matrix(y_test,final_preds))

# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(stack, file)

print("Model saved as model.pkl")
