# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint
import pickle

# Example: assume you have data ready
# X_train, X_test, y_train, y_test
import pandas as pd 
data=pd.read_csv(r'C:\Users\purva\OneDrive\Desktop\NASA Hackathon\Grasping-NASA-Hackathon-\Model_Building-Backend\Refined Dataset\CandidateandFP\kepler_dataset_refined_dummy.csv')
X=data.drop(columns='Classification',axis=0)
Y=data['Classification']
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=3)
# 1️⃣ Define a pipeline: scaling → random forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# 2️⃣ Define hyperparameter distributions for Randomized Search
param_distributions = {
    'rf__n_estimators': randint(100, 500),
    'rf__max_depth': randint(5, 50),
    'rf__min_samples_split': randint(2, 10),
    'rf__min_samples_leaf': randint(1, 5),
    'rf__bootstrap': [True, False]
}

# 3️⃣ Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,            # number of random combinations to try
    cv=5,                 # 5-fold cross-validation
    scoring='accuracy',   # use 'r2' or 'neg_mean_squared_error' for regression
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# 4️⃣ Fit the model
random_search.fit(train_x,train_y)
# 5️⃣ Get the best pipeline
best_model = random_search.best_estimator_

print("\n✅ Best Parameters found:")
print(random_search.best_params_)

# 6️⃣ Evaluate on test data
y_pred = best_model.predict(test_x)

print("\n🎯 Test Accuracy:", accuracy_score(test_y, y_pred))
print("\nClassification Report:\n", classification_report(test_y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(test_y, y_pred))

with open("model2.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Model saved as model.pkl")
