import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Read the cleaned data
df = pd.read_csv('data/heart_cleaned.csv')

# X/y split
X = df.drop("target", axis=1)
y = df["target"]

np.random.seed(42)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid for Logistic Regression
log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}

# RandomizedSearchCV
rs_log_reg = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True,
    random_state=42
)
rs_log_reg.fit(x_train, y_train)

# Output results
print("Best Parameters:", rs_log_reg.best_params_)
print("Test Accuracy:", rs_log_reg.score(x_test, y_test))
print("First test sample features:\n", x_test.iloc[0])
print("Prediction for first test sample:", rs_log_reg.predict([x_test.iloc[0]]))
