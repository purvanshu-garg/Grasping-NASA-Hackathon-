# Grasping-NASA-Hackathon: Exoplanet Classification

## Website & Usage Guide

This project features a locally hosted web application for exoplanet classification.

To run the website:
1.  Navigate to the **API development folder**.
2.  Run `app.py` in your terminal.
3.  The website will launch in your browser (e.g., Chrome).

**Note on Models:** We currently use an **Ensemble Model**. You can easily experiment with other models by changing the model option. Be aware that you may need to adjust code in `app.py` if your new model requires features like a `StandardScaler`.

### Key Features and Known Issues:

1.  **Prediction Options:** The website offers two ways to get predictions:
    * **Single Datapoint:** Enter individual feature values.
    * **File Upload:** Upload a CSV file and get a table of predictions.
2.  **Single Datapoint Bug:** After a single prediction, you **must refresh the page** to check a new data point due to an unaddressed bug. We welcome any suggestions or updates to fix this!
3.  **Tabular Data Download:** The option to download the prediction table is **yet to be added**. For now, please copy the table content or use browser inspect tools to extract and paste into an Excel/spreadsheet file (`.xlsx`).

---

## Model Performance: Ensemble Classifier

Our classification engine is an **Ensemble Model** combining the strengths of a Neural Network and a Random Forest. The goal is to classify stellar objects as either **Exoplanet Candidates** (class **1**) or **False Positives** (class **0**).

## Model Architecture

The final model is a **Stacking or Weighted Average Ensemble** combining predictions from:
1.  An **Optimized Random Forest Classifier** 
2.  A **Tuned Neural Network** 

---

## Final Performance Metrics

The model's performance on the held-out test set is summarized below:

### Overall Accuracy
| Metric | Value |
| :--- | :--- |
| **Ensemble Accuracy** | **0.8346** (**83.46%**) |
| **Neural Network Cross-Validation Accuracy** | 0.8343 |

### Ensemble Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** (False Positive) | **0.81** | 0.71 | 0.76 | 1122 |
| **1** (Exoplanet Candidate) | **0.85** | **0.90** | **0.87** | 1968 |
| **Macro Average** | 0.83 | 0.81 | 0.82 | 3090 |
| **Weighted Average** | 0.83 | 0.83 | 0.83 | 3090 |

### Key Observations:

* **High Recall for Class 1 (Exoplanets):** The model achieved a remarkable **Recall of 0.90** for Class 1. This is crucial as it means the model is highly effective at minimizing **False Negatives** (missing a real exoplanet), correctly identifying **90%** of the actual exoplanet candidates.
* **Strong Precision for Class 1:** A **Precision of 0.85** for Class 1 indicates that when the model labels an object as an exoplanet, it is correct **85%** of the time.

---

## Hyperparameters

The optimal settings determined through cross-validation for the **Random Forest** component were:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `rf__max_depth` | **None** | Allows maximum depth expansion (until leaves are pure or below `min_samples_split`). |
| `rf__min_samples_split` | **10** | Minimum number of samples required to split an internal node. |
| `rf__n_estimators` | **300** | The number of trees in the forest. |
