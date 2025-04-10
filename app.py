import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("rf_model.pkl")

# App Title
st.set_page_config(page_title="Data Analyst Salary Predictor", layout="centered")
st.title("Salary Predictor-19")
st.markdown("Predict the salary of a data analyst based on job-related features.")

# Input features
st.header("📥 Enter Job Information")

Rating = st.slider("⭐ Company Rating", min_value=1.0, max_value=5.0, step=0.1)
job_title_cleaned = st.selectbox("💼 Job Type", ["Full-time", "Part-time", "Internship", "Contract"])
job_city = st.selectbox("📍 Job Location", ["Remote", "Bangalore", "Mumbai", "New Delhi", "Others"])
Company_Name = st.text_input("🏢 Company Name")
Salary_Range = st.text_input("💰 Salary Range")

# Convert inputs into a DataFrame (format must match training set)
input_df = pd.DataFrame({
    "Company Rating": [Rating],
    "Job Title": [job_title_cleaned],
    "Location": [job_city],
    # Replace the following placeholders with actual input variables or remove them if not used
    "Company_Name": ["Company_Name"],
    "Salary_Range": ["Salary_Range"]
})

# TODO: Apply the same preprocessing (encoding, etc.) used during training here
# For now, assume the model handles it internally or you manually apply encoders

# Predict
if st.button("🔮 Predict Salary"):
    try:
        prediction = model.predict(input_df)
        st.success(f"💰 Estimated Salary: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error("⚠️ Error making prediction. Please check your inputs or preprocessing.")
        st.text(str(e))

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Khushi | Powered by Random Forest + Streamlit")

