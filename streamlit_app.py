# Bank Marketing - Machine Learning Classification Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import optuna
from imblearn.over_sampling import SMOTE

# 1. Load Data
df = pd.read_csv("bank-additional.csv", sep=';')

# 2. Clean 'unknown' values
columns_to_fill = ['job', 'marital', 'education', 'housing', 'loan']
for column in columns_to_fill:
    df[column] = df[column].replace('unknown', df[column].mode()[0])

# 3. Preprocessing
X = df.drop(columns='y')
y = df['y']

# 3.a Feature Engineering
X['campaign_per_previous'] = X['campaign'] / (X['previous'] + 1)
X['contact_month_combo'] = X['contact'] + '_' + X['month']
X['loan_and_housing'] = X['loan'] + '_' + X['housing']

# Re-encode new categorical columns
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Encode target
y_encoded = LabelEncoder().fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE for class balance
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

# Train Final XGBoost Model with Best Parameters
xgb_final_params = {
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.07216874784717071,
    'subsample': 0.988839400807138,
    'colsample_bytree': 0.8801296358249565,
    'gamma': 3.532791449428222,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

final_model = XGBClassifier(**xgb_final_params)
final_model.fit(X_train_balanced, y_train_balanced)
probs_final = final_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs_final)
f1s = 2 * (precisions * recalls) / (precisions + recalls)
best_threshold = thresholds[np.argmax(f1s)]

# STREAMLIT UI
st.title("Term Deposit Subscription Prediction")
st.write("Enter client information to predict if they will subscribe to a term deposit:")

input_data = {}
for col in X.columns:
    if col in numerical_cols:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    else:
        input_data[col] = st.selectbox(f"{col}", options=list(df[col].unique()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    proba = final_model.predict_proba(input_df)[:, 1]
    prediction = (proba > best_threshold).astype(int)[0]

    st.write("\n### Result:")
    if prediction == 1:
        st.success("Client is likely to subscribe to a term deposit.")
    else:
        st.warning("Client is unlikely to subscribe to a term deposit.")
