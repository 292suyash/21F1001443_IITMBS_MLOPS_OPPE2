import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set MLflow experiment
mlflow.set_experiment("heart_disease_experiment")

# Load and preprocess data
df = pd.read_csv('data/heart_cleaned.csv')

# One-hot encode 'gender' and 'target' if not already numeric
if df['gender'].dtype == 'O':
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
if df['target'].dtype == 'O':
    df['target'] = df['target'].map({'yes': 1, 'no': 0})

# Drop rows with missing values (should be handled in prepare_data.py, but just in case)
df = df.dropna()

# Features and target
X = df.drop(['target', 'sno'], axis=1)
y = df['target']

# Train/test split
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}

# Randomized search
rs_log_reg = RandomizedSearchCV(
    LogisticRegression(max_iter=200),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=1
)
rs_log_reg.fit(X_train, y_train)
best_params = rs_log_reg.best_params_
best_score = rs_log_reg.score(X_test, y_test)

# MLflow logging
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)
    mlflow.sklearn.log_model(rs_log_reg.best_estimator_, "model")
    # Save model locally for Docker/serving
    mlflow.sklearn.save_model(rs_log_reg.best_estimator_, "model")

print(f"Best hyperparameters: {best_params}")
print(f"Test accuracy: {best_score:.4f}")
