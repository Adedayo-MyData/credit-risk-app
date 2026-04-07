#!/usr/bin/env python
# coding: utf-8

# In[9]:


import joblib
import pandas as pd
from preprocessed import preprocess_input

# Load saved objects
xgb = joblib.load("xgb.pkl")

def predict_risk(input_dict):
    
    data = pd.DataFrame([input_dict])
    data = preprocess_input(data)

    expected_cols = model.feature_names_in_
    data = data.reindex(columns=expected_cols, fill_value=0)

    prediction = xgb.predict(data)[0]
    probability = xgb.predict_proba(data)[0][1]
    
    risk_level, decision = make_decision(probability)

    # SHAP values
    shap_values = explainer(data)
    
    return prediction, probability, risk_level, decision, shap_values, data

def make_decision(probability):
    
    if probability < 0.30:
        return "LOW RISK", "APPROVE"
    
    elif probability < 0.60:
        return "MEDIUM RISK", "REVIEW"
    
    else:
        return "HIGH RISK", "REJECT"


# In[ ]:





# In[ ]:




