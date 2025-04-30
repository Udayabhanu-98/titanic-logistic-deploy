
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Predictor (Final Features)")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ["male", "female"])
title = st.selectbox("Title", ["Mr", "Miss", "Master", "Mme", "Ms", "Dr", "Lady", "Col", "Capt"])

# Encode sex
sex_male = 1 if sex == "male" else 0

# Encode title only if it's in trained features
expected_title_cols = ['Title_Master', 'Title_Miss', 'Title_Mme', 'Title_Ms']
title_map = {
    "Master": "Title_Master",
    "Miss": "Title_Miss",
    "Mme": "Title_Mme",
    "Ms": "Title_Ms"
}
title_encoded = {col: 0 for col in expected_title_cols}
if title in title_map:
    title_encoded[title_map[title]] = 1

# Scale age and parch
scaled_vals = scaler.transform(pd.DataFrame([[age, parch]], columns=["Age", "Parch"]))
age_scaled, parch_scaled = scaled_vals[0]

# Construct input
input_dict = {
    "Pclass": pclass,
    "Age": age_scaled,
    "Parch": parch_scaled,
    "Sex_male": sex_male,
    **title_encoded
}

# Enforce column order
final_input = pd.DataFrame([input_dict])
final_input = final_input[['Pclass', 'Age', 'Parch', 'Sex_male', 'Title_Master', 'Title_Miss', 'Title_Mme', 'Title_Ms']]

# Predict
if st.button("Predict Survival"):
    pred = model.predict(final_input)[0]
    result = "🟩 Survived" if pred == 1 else "🟥 Did Not Survive"
    st.subheader("Prediction Result")
    st.success(result)
