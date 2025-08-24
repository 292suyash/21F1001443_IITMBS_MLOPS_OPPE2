import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/heart_cleaned.csv')
X = df.drop("target", axis=1)
y = df["target"]

model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X, show=False)
plt.savefig('shap_summary.png')

# Print top features
shap_importance = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
print("Top features by SHAP importance:")
print(shap_importance)
