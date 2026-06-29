import joblib
import pandas as pd

# Load saved model
model = joblib.load("models/iris_model.pkl")

# Take input from user
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

# Create DataFrame
sample = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

# Predict species
prediction = model.predict(sample)

# Display result
print("\nPredicted Species:", prediction[0])