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

st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

h1{
    color:#1565C0;
    text-align:center;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#1565C0;
    color:white;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0D47A1;
}

div[data-testid="stSidebar"]{
    background:#E3F2FD;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/iris_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.markdown(
"""
# 🌸 Iris Flower Classification

### Predict Iris Species using Machine Learning

---
"""
)
st.write("Predict the species of an Iris flower using Machine Learning.")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📌 Project Information")

st.sidebar.write("Developer : Vaibhav")

st.sidebar.write("Branch : Data Science & AI")

st.sidebar.write("Dataset : Iris Dataset")

st.sidebar.write("Algorithms:")

st.sidebar.write("✔ Logistic Regression")
st.sidebar.write("✔ KNN")
st.sidebar.write("✔ Decision Tree")
st.sidebar.write("✔ Random Forest")

st.sidebar.success("Best Accuracy : 100%")
# -----------------------------
# User Inputs
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length",
        value=5.1
    )

    petal_length = st.number_input(
        "Petal Length",
        value=1.4
    )

with col2:

    sepal_width = st.number_input(
        "Sepal Width",
        value=3.5
    )

    petal_width = st.number_input(
        "Petal Width",
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

    st.success(f"🌼 Prediction : {prediction[0]}")

    st.balloons()

st.divider()

st.caption("Made with ❤️ using Streamlit")