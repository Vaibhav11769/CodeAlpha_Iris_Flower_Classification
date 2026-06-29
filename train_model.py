import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("dataset/Iris.csv")

# Remove unnecessary column
df.drop("Id", axis=1, inplace=True)

# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("Species", axis=1)
y = df["Species"]

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 50)
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("=" * 50)

# -----------------------------
# Models
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

print("\nModel Accuracy\n")

best_accuracy = 0
best_model_object = None

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_object = model

    print(f"{name:<22}: {accuracy:.4f}")

    print(f"\n{name} - Classification Report")
    print(classification_report(y_test, prediction))

    cm = confusion_matrix(y_test, prediction)

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
    )

    disp.plot(cmap="Blues")

    plt.title(name)

    plt.show()

    print("-"*60)

best_model = max(results, key=results.get)

joblib.dump(
    best_model_object,
    "models/iris_model.pkl"
)

print("\nModel Saved Successfully!")

print("\nBest Model :", best_model)
print("Accuracy   :", round(best_accuracy * 100, 2), "%")