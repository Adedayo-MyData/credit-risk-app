#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

def preprocess_input(data):
    
    # --- Feature Engineering ---
    data['monthly_payment_est'] = data['loan_amnt'] / 12
    data['income_per_year_of_emp'] = data['person_income'] / (data['person_emp_length'] + 1)
    data['interest_income_ratio'] = data['loan_int_rate'] / (data['person_income'] + 1)
    data['credit_exp_ratio'] = data['cb_person_cred_hist_length'] / data['person_age']
    
    # --- Encoding ---
    data['loan_grade'] = data['loan_grade'].map({
        'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7
    })
    
    data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({
        'Y':1, 'N':0
    })
    
    # --- One-hot encoding ---
    data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], drop_first=True)
    
    return data


# In[ ]:




