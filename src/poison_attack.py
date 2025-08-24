import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/heart_cleaned.csv')
X = df.drop("target", axis=1)
y = df["target"]

# Flip labels for 10% of data
y_poisoned = y.copy()
n_flip = int(0.1 * len(y))
y_poisoned.iloc[:n_flip] = 1 - y_poisoned.iloc[:n_flip]

x_train, x_test, y_train, y_test = train_test_split(X, y_poisoned, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy after poisoning: {acc:.4f}")
