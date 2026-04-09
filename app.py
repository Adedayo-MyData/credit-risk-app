#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import streamlit as st
from predict import predict_risk
import shap
import matplotlib.pyplot as plt


st.title("Credit Risk Prediction System")

# --- User Inputs ---
age = st.number_input("Age", 18, 100)
income = st.number_input("Annual Income")
home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
emp_length = st.number_input("Employment Length (years)")
intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL"])
grade = st.selectbox("Loan Grade", list("ABCDEFG"))
loan_amount = st.number_input("Loan Amount")
interest_rate = st.number_input("Interest Rate")
loan_to_income_pencentage = st.number_input("Loan to Income %")
default = st.selectbox("Previous Default", ["Y", "N"])
cred_hist = st.number_input("Credit History Length")

# --- Prediction ---
if st.button("Predict Risk"):
    
    input_data = {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": emp_length,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": loan_amount,
        "loan_int_rate": interest_rate,  
        "loan_percent_income": loan_to_income_pencentage,
        "cb_person_default_on_file": default,
        "cb_person_cred_hist_length": cred_hist
    }
            
    pred, prob, risk, decision, shap_values, data = predict_risk(input_data)

    shap_df = pd.DataFrame({
    "feature": data.columns,
    "impact": shap_values.values[0],
    "value": data.iloc[0]
    })

    # Remove confusing features
    shap_df = shap_df[
        ~((shap_df['value'] == 0) & (shap_df['feature'].str.contains("loan_intent|person_home")))
    ]
    
    feature_names_map = {
    "person_income": "Income",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate",
    "loan_percent_income": "Loan vs Income Ratio",
    "income_per_year_of_emp": "Income Stability",
    "interest_income_ratio": "Interest Burden",
    "credit_exp_ratio": "Credit Experience",
    "loan_grade": "Loan Grade"
    }

    shap_df['feature'] = shap_df['feature'].map(feature_names_map).fillna(shap_df['feature'])
    
    top = shap_df.reindex(
    shap_df.impact.abs().sort_values(ascending=False).index
    ).head(3)
    
    st.write(f"### Risk Score: {prob:.2f}")

    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")
    
    
    st.write("### 🔍 Why this decision was made")

    for _, row in top.iterrows():
        if row['impact'] > 0:
            st.write(f"⚠️ {row['feature']} increased the risk")
        else:
            st.write(f"✅ {row['feature']} reduced the risk")
            
    st.write("### 📌 Decision Summary")

    if decision == "APPROVE":
        st.success("This loan is likely safe because key financial indicators are strong.")

    elif decision == "REVIEW":
        st.warning("This loan has mixed risk signals and should be reviewed manually.")

    else:
        st.error("This loan is risky due to unfavorable financial indicators.")
    
    
    


# In[ ]:




