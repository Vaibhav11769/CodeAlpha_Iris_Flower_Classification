import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset/Iris.csv")

# Remove Id column
df.drop("Id", axis=1, inplace=True)

print("=" * 50)
print("Dataset Statistics")
print("=" * 50)
print(df.describe())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nSpecies Count")
print(df["Species"].value_counts())

print("\nUnique Species:")
print(df["Species"].unique())

print("\nDataset Shape:")
print(df.shape)

# Count Plot
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Species")
plt.title("Species Distribution")
plt.show()

# Pair Plot
sns.pairplot(df, hue="Species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    df.drop("Species", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.show()