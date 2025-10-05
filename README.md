# Grasping-NASA-Hackathon

Website :
This is locally hosted website
To run the website use API development folder, run app.py in your terminal and then website will run in chrome.
We have used ensemble model you can try other models also. This line can be changed in model option and thus any other model can be run. You might have to add some other features in app.py if you use scaler etc for training.
1) Consists of Two options either data a single data point and get the result or you can upload the file and then you will get a table showing the predictions corresponding to your datapoint.
2) For Single Datapoint there is prediction but for checking new data point you need to refresh the page since there was a bug which could not be fixed. Any suggested changes or updates will be highly appreciated.
3) For tabular data also downloadable version is to be added. You can copy or using inspect take the table and paste in xlsx file for now.

Model:
Now Talking about our model it is based on Ensemble model ( Neural Network and Random Forest ). Statistics of this model is given below :

# Ensemble Exoplanet Classifier Model Report

This README summarizes the final performance of the ensemble model used for classifying exoplanet candidates (likely exoplanets, class **1**) versus false positives (class **0**). The model utilizes a combination of a optimized Random Forest and a Neural Network.

## 🚀 Model Architecture

The final model is an **Ensemble Model** (Stacking or Weighted Average), combining predictions from:
1.  **Optimized Random Forest Classifier**
2.  **Tuned Neural Network**

## 📊 Final Performance Metrics

The model was evaluated on a held-out test set, and its performance is reported below:

### Overall Accuracy
| Metric | Value |
| :--- | :--- |
| **Ensemble Accuracy** | **0.8346** (83.46%) |
| **Neural Network Cross-Validation Accuracy** | 0.8343 |

### Ensemble Classification Report

The following metrics detail the precision, recall, and F1-score for each class:

| Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** (False Positive) | 0.81 | 0.71 | 0.76 | 1122 |
| **1** (Exoplanet Candidate) | 0.85 | 0.90 | 0.87 | 1968 |
| **Macro Average** | 0.83 | 0.81 | 0.82 | 3090 |
| **Weighted Average** | 0.83 | 0.83 | 0.83 | 3090 |

### Key Observations:

* **High Recall for Class 1 (Exoplanets):** The model achieved a **Recall of 0.90** for Class 1 (Exoplanet Candidate). This means the model successfully identified 90% of the actual exoplanet candidates in the test set, making it highly effective at minimizing **False Negatives** (failing to identify a real exoplanet).
* **Strong Precision for Class 1:** The **Precision of 0.85** for Class 1 indicates that when the model predicts an object is an exoplanet, it is correct 85% of the time.

## ⚙️ Hyperparameters


The optimal hyperparameters found during the cross-validation for the Random Forest component were:

| Parameter | Value |
| :--- | :--- |
| `rf__max_depth` | None (allows nodes to expand until all leaves are pure or contain fewer than `min_samples_split` samples) |
| `rf__min_samples_split` | 10 |
| `rf__n_estimators` | 300 |




