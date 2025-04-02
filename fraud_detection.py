import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline

# Load dataset
data_path = '/Users/saloni/Desktop/College/Project/frauddetection_R/creditcard.csv'  # Change path if running locally
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError("Dataset not found! Check the path.")

# Handle missing values
data.fillna(0, inplace=True)

# Display dataset info
print("Dataset Overview:")
print(data.describe())
print(f"Valid transactions: {len(data[data['Class'] == 0])}")
print(f"Fraud transactions: {len(data[data['Class'] == 1])}")

# Splitting dataset
y = data['Class']
X = data.drop(columns=['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, title="Model Performance"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"{title}:")
    print(classification_report(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_mat)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Model 1: Random Forest (Baseline)
rf_baseline = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0, max_depth=10)
train_and_evaluate(rf_baseline, X_train, X_test, y_train, y_test, "Random Forest - Baseline")

# Model 2: Random Forest with Undersampling
undersampler = RandomUnderSampler()
pipeline_under = Pipeline([('undersample', undersampler), ('RF', rf_baseline)])
train_and_evaluate(pipeline_under, X_train, X_test, y_train, y_test, "Random Forest - Undersampling")

# Model 3: Random Forest with Oversampling
oversampler = RandomOverSampler()
pipeline_over = Pipeline([('oversample', oversampler), ('RF', rf_baseline)])
train_and_evaluate(pipeline_over, X_train, X_test, y_train, y_test, "Random Forest - Oversampling")

# Model 4: Random Forest with SMOTE
smote_sampler = SMOTE(sampling_strategy='auto', random_state=0)
pipeline_smote = Pipeline([('SMOTE', smote_sampler), ('RF', rf_baseline)])
train_and_evaluate(pipeline_smote, X_train, X_test, y_train, y_test, "Random Forest - SMOTE")
