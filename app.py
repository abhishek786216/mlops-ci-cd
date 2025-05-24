import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model (in real app you’d load a saved model)
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("Iris Species Predictor")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", 5.0, 8.0, 5.0)
sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction function
def predict_species(features):
    features = np.array(features).reshape(1, -1)
    pred_idx = model.predict(features)[0]
    return iris.target_names[pred_idx]

if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict_species(features)
    st.success(f"The predicted iris species is: **{prediction}**")
