import pandas as pd

df = pd.read_csv("symbipredict_2022.csv")

unique_values = df['prognosis'].unique()

# Convert to list (if needed)

print(unique_values)