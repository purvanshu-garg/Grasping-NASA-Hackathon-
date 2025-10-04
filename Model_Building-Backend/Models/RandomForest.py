import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)  
    classification_report(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }


def random_forest(file_name):
    data=pd.read_csv(file_name)
    X=data.drop(columns=['Classification'],axis=0)
    Y=data['Classification']
    train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=3)
    


