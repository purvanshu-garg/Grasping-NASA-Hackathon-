
# ===========================
# Exoplanet Classification Ensemble Model
# Random Forest + Neural Network (MLP)
# With GridSearchCV + Cross-Validation
# ===========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier


np.random.seed(42)

data=pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\kepler_tess_combined_dummy.csv')

X = data.drop('Classification', axis=1)
y = data['Classification']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ======================================
# 2️⃣ Random Forest with GridSearchCV
# ======================================
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

rf_params = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid=rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)
print("Best Random Forest Params:", rf_grid.best_params_)

best_rf = rf_grid.best_estimator_

# ======================================
# 3️⃣ Neural Network (MLP) with CV
# ======================================
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32),
                          activation='relu',
                          solver='adam',
                          max_iter=500,
                          random_state=42))
])

# Perform K-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mlp_cv_scores = cross_val_score(mlp_pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print("Neural Network CV Accuracy:", mlp_cv_scores.mean())

# Train the final MLP on full training set
mlp_pipeline.fit(X_train, y_train)

# ======================================
# 4️⃣ Combine into Voting Ensemble
# ======================================
ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('mlp', mlp_pipeline)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# ======================================
# 5️⃣ Evaluate
# ======================================
print("\n=== Ensemble Model Report ===")
print(classification_report(y_test, y_pred))
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))

import pickle
with open("ensemble.pkl", "wb") as file:
    pickle.dump(ensemble, file)
print("Model saved as model.pkl")