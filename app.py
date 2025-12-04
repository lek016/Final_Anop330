st.markdown("Predict your invitation using our top 5 features!")

reunion_years_out = st.number_input(
    "How many years out from graduation are they? 🎓",
    min_value=0,
    max_value=80,
    value=5
)

peer = st.selectbox("Did a friend refer you? 🫂", ["Yes", "No"])
volunteer = st.selectbox("Do you volunteer in the Bucknell community? ❤️", ["Yes", "No"])
greek = st.selectbox("Were you in Greek life? 🏠", ["Yes", "No"])
engineering_bachelor = st.selectbox("Did you obtain a Bachelor's Degree in Engineering? 📏", ["Yes", "No"])

# Create input dataframe with EXACT column names
input_data = pd.DataFrame([{
    'Reunion_Years_Out': int(reunion_years_out),
    'Peer': 1 if peer == "Yes" else 0,
    'Volunteer': 1 if volunteer == "Yes" else 0,
    'Greek?': 1 if greek == "Yes" else 0,
    'Engineering_Bachelor': 1 if engineering_bachelor == "Yes" else 0
}])

if st.button("🎉 Predict"):
    prob = model.predict_proba(input_data)[0][1]
    pred = 1 if prob > Threshold else 0

    if pred == 1:
        st.success(f"✅ Yes! Likely to accept the invitation ({prob*100:.1f}% confidence)")
    else:
        st.error(f"❌ No. Unlikely to accept the invitation ({(1-prob)*100:.1f}% confidence)")
    if pred == 1:
        st.success(f"✅ Yes! Likely to accept the invitation ({prob*100:.1f}% confidence)")
    else:
        st.error(f"❌ No. Unlikely to accept the invitation ({(1-prob)*100:.1f}% confidence)")
