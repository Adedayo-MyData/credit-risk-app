#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

    st.write(f"### Risk Score: {prob:.2f}")

    if prob < 0.3:
        st.success("Low Risk")
    elif prob < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")
    
    st.write("### Model Explanation")

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)


# In[ ]:




