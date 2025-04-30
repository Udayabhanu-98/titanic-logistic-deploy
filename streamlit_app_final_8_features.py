
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")  # This should contain StandardScaler used for Age and Parch

st.title("🚢 Titanic Survival Predictor (Final Features)")
st.write("Enter passenger details to predict survival.")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
title = st.selectbox("Title", ["Master", "Miss", "Mme", "Ms"])

# Manual one-hot encoding
input_dict = {
    "Pclass": pclass,
    "Sex_male": 1 if sex == "male" else 0,
    "Title_Master": 1 if title == "Master" else 0,
    "Title_Miss": 1 if title == "Miss" else 0,
    "Title_Mme": 1 if title == "Mme" else 0,
    "Title_Ms": 1 if title == "Ms" else 0
}

# Add Age and Parch, then scale them
input_df = pd.DataFrame([{
    "Age": age,
    "Parch": parch
}])
scaled_vals = scaler.transform(input_df)
input_dict["Age"] = scaled_vals[0][0]
input_dict["Parch"] = scaled_vals[0][1]

# Final input dataframe
final_input = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Survival"):
    pred = model.predict(final_input)[0]
    result = "🟩 Survived" if pred == 1 else "🟥 Did Not Survive"
    st.subheader("Prediction")
    st.success(result)
