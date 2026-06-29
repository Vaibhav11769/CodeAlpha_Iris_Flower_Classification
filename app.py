import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/iris_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🌸 Iris Flower Classification")
st.write("Predict the species of an Iris flower using Machine Learning.")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About Project")

st.sidebar.info("""
This project classifies Iris flowers into three species using Machine Learning.

Models Compared:
- Logistic Regression
- KNN
- Decision Tree
- Random Forest

Best Accuracy:
100%
""")

# -----------------------------
# User Inputs
# -----------------------------

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    value=0.2
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict"):

    sample = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )

    prediction = model.predict(sample)

    st.success(f"🌼 Predicted Species: {prediction[0]}")

st.divider()

st.caption("Made with ❤️ using Streamlit")