import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained XGBoost model
model = joblib.load("optimized_xgboost_model.pkl")

# Set the threshold manually (from your notebook)
best_threshold = 0.472  # örnek bir değer, senin en iyi threshold değerine göre güncelle

# Define columns manually (from your notebook)
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'contact_month_combo', 'loan_and_housing']
numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign_per_previous']

all_columns = numerical_cols + categorical_cols

# LabelEncoder mappings (manually re-fit with training data in real scenario)
# To keep this self-contained, define dummy mappings for demo purposes
label_mappings = {
    'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'],
    'marital': ['divorced', 'married', 'single'],
    'education': ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree'],
    'default': ['no', 'yes'],
    'housing': ['no', 'yes'],
    'loan': ['no', 'yes'],
    'contact': ['cellular', 'telephone'],
    'month': ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
    'poutcome': ['failure', 'nonexistent', 'success'],
    'contact_month_combo': ['cellular_may', 'telephone_jul', 'cellular_aug'],  # örnek sınıflar
    'loan_and_housing': ['no_no', 'no_yes', 'yes_no', 'yes_yes']
}

label_encoders = {}
for col, classes in label_mappings.items():
    le = LabelEncoder()
    le.fit(classes)
    label_encoders[col] = le

# Dummy scaler for numerical columns (in real app, fit this on training data)
scaler = StandardScaler()
scaler.fit(np.zeros((1, len(numerical_cols))))  # Dummy fit to prevent error

# Streamlit UI
st.set_page_config(page_title="Term Deposit Predictor", layout="centered")
st.title("📊 Term Deposit Subscription Prediction")
st.write("Provide client information below:")

input_data = {}

with st.form("client_form"):
    for col in numerical_cols:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    for col in categorical_cols:
        input_data[col] = st.selectbox(f"{col}", options=label_mappings[col])
    submitted = st.form_submit_button("Predict")

if submitted:
    df_input = pd.DataFrame([input_data])

    # Label encode
    for col in categorical_cols:
        df_input[col] = label_encoders[col].transform(df_input[col])

    # Scale numericals
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])

    # Predict
    proba = model.predict_proba(df_input)[:, 1][0]
    prediction = int(proba > best_threshold)

    st.markdown("---")
    st.metric("Probability", f"{proba:.2%}")

    if prediction == 1:
        st.success("✅ The client is **likely** to subscribe to a term deposit.")
    else:
        st.warning("⚠️ The client is **unlikely** to subscribe to a term deposit.")
