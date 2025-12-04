import streamlit as st
import joblib
import pandas as pd

loaded = joblib.load("XGmodel_0.2.pkl")
model = loaded["model"]
Threshold = loaded["threshold"]

st.set_page_config(page_title="🎉 Reunion Prediction App", layout="centered")

st.image("Bucknell.jpeg", use_column_width=True)

# Title
st.title("Reunion Invitation Acceptance Prediction App")
st.markdown("Predict whether someone will accept your invitation to the reunion using our top 5 features!")

# Graduation year as slider
grad_year = st.slider("What year did you graduate? 🎓", min_value=1950, max_value=2025, value=2019)
current_year = 2025
reunion_years_out = current_year - grad_year

peer = st.selectbox("Did a friend refer you? 🫂", ["Yes", "No"])
volunteer = st.selectbox("Do you volunteer in the Bucknell community? ❤️", ["Yes", "No"])
greek = st.selectbox("Were you in Greek life? 🏠", ["Yes", "No"])
engineering_bachelor = st.selectbox("Did you obtain a Bachelor's Degree in Engineering? 📏", ["Yes", "No"])

# FIXED: Convert inputs to model format using BOOLEANS (not integers)
input_data = pd.DataFrame([{
    'Reunion_Years_Out': int(reunion_years_out),
    'Peer': True if peer == "Yes" else False,
    'Volunteer': True if volunteer == "Yes" else False,
    'Greek?': True if greek == "Yes" else False,
    'Engineering_Bachelor': True if engineering_bachelor == "Yes" else False
}])

# Predict
if st.button("🎉 Predict"):
    prob = model.predict_proba(input_data)[0][1]
    pred = 1 if prob > Threshold else 0

    if pred == 1:
        st.success(f"✅ Yes! Likely to accept your invitation ({prob*100:.1f}% confidence)")
    else:
        st.error(f"❌ No. Unlikely to accept your invitation ({(1-prob)*100:.1f}% confidence)")
