import pandas as pd

df = pd.read_csv('data/heart_cleaned.csv')

# Since 'target' is now numeric: 1 for 'yes', 0 for 'no'
n_yes = (df['target'] == 1).sum()
n_no = (df['target'] == 0).sum()

print(f"Count of 'yes': {n_yes}, Count of 'no': {n_no}")
