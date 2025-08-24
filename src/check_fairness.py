import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/heart_cleaned.csv')
X = df.drop("target", axis=1)
y = df["target"]
sensitive = X['gender'] if 'gender' in X.columns else None

model = LogisticRegression(max_iter=200, solver='liblinear', C=30)
model.fit(X, y)
y_pred = model.predict(X)

if sensitive is not None:
    metric_frame = MetricFrame(metrics=accuracy_score,
                               y_true=y,
                               y_pred=y_pred,
                               sensitive_features=sensitive)
    print("Fairness metrics by gender:")
    print(metric_frame.by_group)
else:
    print("No sensitive feature found for fairness check.")
