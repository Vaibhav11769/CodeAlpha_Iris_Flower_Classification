import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Iris.csv")

print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Distribution:")
print(df["Species"].value_counts())

print("\nDataset Info:")
print(df.info())