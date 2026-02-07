import streamlit as st
import joblib
import numpy as np

# Page config (VERY IMPORTANT for modern look)
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# Load model & scaler
Model = joblib.load("Model/loan_prediction_model.pkl")
scaler = joblib.load("Model/scaler.pkl")

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>🏦 Loan Approval Prediction System</h1>
    <p style='text-align: center; font-size:18px; color: grey;'>
    Smart AI-based loan eligibility checker
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
st.sidebar.header("📋 Applicant Details")

no_of_dependents = st.sidebar.number_input("👨‍👩‍👧 Dependents", 0, 10, 1)
education = st.sidebar.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("💼 Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("💰 Annual Income", 0)
loan_amount = st.sidebar.number_input("🏷️ Loan Amount", 0)
loan_term = st.sidebar.number_input("⏳ Loan Term (Years)", 1)
cibil_score = st.sidebar.number_input("📊 CIBIL Score", 300, 900)
residential_assets_value = st.sidebar.number_input("🏠 Residential Assets", 0)
commercial_assets_value = st.sidebar.number_input("🏢 Commercial Assets", 0)
luxury_assets_value = st.sidebar.number_input("🚗 Luxury Assets", 0)
bank_asset_value = st.sidebar.number_input("🏦 Bank Assets", 0)

# Encoding (same logic, cleaned)
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# ---------- MAIN CONTENT ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📌 Applicant Summary")
    st.write(f"""
    • **Dependents:** {no_of_dependents}  
    • **Education:** {'Graduate' if education == 1 else 'Not Graduate'}  
    • **Self Employed:** {'Yes' if self_employed == 1 else 'No'}  
    • **Annual Income:** ₹{income_annum}  
    • **Loan Amount:** ₹{loan_amount}  
    • **CIBIL Score:** {cibil_score}
    """)

with col2:
    st.subheader("📈 Key Indicators")
    st.metric("Income", f"₹{income_annum}")
    st.metric("Loan Amount", f"₹{loan_amount}")
    st.metric("CIBIL Score", cibil_score)

st.markdown("---")

# ---------- PREDICTION ----------
if st.button("🔍 Predict Loan Status", use_container_width=True):
    input_data = np.array([[  
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    input_scaled = scaler.transform(input_data)

    probability = Model.predict_proba(input_scaled)[0][1]

    THRESHOLD = 0.35
    prediction = 1 if probability >= THRESHOLD else 0

    st.markdown("## 🧾 Prediction Result")
    st.write(f"**Risk Probability:** {probability:.2f}")
    # st.write(f"**Decision Threshold:** {THRESHOLD}")

    if prediction == 1:
        st.error("❌ **Loan Rejected** ⚠️")
        st.warning("High risk of default detected.")
    else:
        st.success("✅ **Loan Approved** 🎉")
        st.balloons()


# ---------- FOOTER ----------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: grey;'>
    Built with using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
