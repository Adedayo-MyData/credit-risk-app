# 💳 Credit Risk Prediction System

A machine learning-powered web application that predicts the likelihood of loan default and provides **interpretable, business-friendly insights** into each decision.

🔗 **Live App:** https://bit.ly/Credit_Risk_App

---

## 📌 Overview

This project delivers an end-to-end **credit risk prediction system** where users input borrower information and receive:

- ✅ Risk Score (Probability of Default)
- ✅ Risk Level (Low / Medium / High)
- ✅ Decision Recommendation (Approve / Review / Reject)
- ✅ Explainable Insights (Why the decision was made)

The system is deployed as an interactive web app using **Streamlit**.

---

## 🚀 Features

- 🔍 Real-time credit risk prediction  
- 🧠 Multiple ML models evaluated (Logistic Regression, Random Forest, XGBoost)  
- ⚖️ Class imbalance handled using **SMOTE**   
- 📊 ROC curve comparison across models  
- 🔎 Explainability using **SHAP**  
- 🏦 Business decision engine (Approve / Review / Reject)  
- 🧾 Human-readable explanations (cleaned SHAP outputs)

---

## 🧠 Model Development

### Models Trained

- Logistic Regression  
- Random Forest  
- XGBoost  

---

### 📊 The Model with the best result was XGBoost, with the performance below:

- ROC-AUC: **~0.95**
- Accuracy: **~0.92**
- Recall (both classes): **~95% and ~80%**
- Cross Validation Score: **~0.94**

---

## ⚙️ Feature Engineering

The model uses engineered features to improve predictive power:

- `monthly_payment_est` → Estimated monthly repayment  
- `income_per_year_of_emp` → Income stability  
- `interest_income_ratio` → Interest burden  
- `credit_exp_ratio` → Credit experience relative to age  

Categorical variables were encoded using:
- Label encoding  
- One-hot encoding  

---

## 🧾 Input Parameters

Users provide:

- Age  
- Annual Income  
- Home Ownership  
- Employment Length  
- Loan Intent  
- Loan Grade  
- Loan Amount  
- Interest Rate  
- Loan-to-Income Ratio  
- Previous Default History  
- Credit History Length  

---

## 🧪 Sample Input

```plaintext
Age: 21
Annual Income: 9600
Home Ownership: OWN
Employment Length: 5
Loan Intent: EDUCATION
Loan Grade: B
Loan Amount: 1000
Interest Rate: 11.14
Loan-to-Income %: 0.1
Previous Default: N
Credit History Length: 2

---

## 📊 Output Interpretation
🔹 Risk Score

Probability between 0 and 1 indicating likelihood of default.

🔹 Risk Levels
Low Risk: < 0.30
Medium Risk: 0.30 – 0.60
High Risk: > 0.60

🔹 Decision Logic
✅ Approve → Low Risk
⚠️ Review → Medium Risk
❌ Reject → High Risk

---

## 🔍 Explainability (SHAP)

The system uses SHAP to explain predictions.

What Users See:
Top 3 factors influencing the decision
Whether each factor increased or reduced risk
Clean, human-readable explanations
Example Output
🔍 Why this decision was made

✅ Loan vs Income Ratio reduced the risk  
⚠️ Interest Burden increased the risk  
⚠️ Income increased the risk  

---

## 🏗️ Project Structure
credit-risk-app/
│
├── app.py               # Streamlit user interface
├── predict.py           # Prediction logic + SHAP explainability
├── preprocess.py        # Data preprocessing & feature engineering
├── xgb.pkl              # Trained ML model         
├── requirements.txt     # Dependencies

[Credit_Risk_App](app.py)
---

## 🛠️ Technologies Used
Python
Scikit-learn
XGBoost
Pandas & NumPy
SHAP (Explainable AI)
Streamlit (Web App Deployment)

---

## 🌍 Deployment

The application is deployed on Streamlit Cloud:

👉 https://bit.ly/Credit_Risk_App

---

## 📈 Future Improvements
Hyperparameter tuning for improved model performance
Advanced feature engineering
Dashboard-style UI enhancements
User authentication and data storage
Model monitoring and drift detection

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome.

---

## 📜 License

This project is intended for educational and demonstration purposes.

---

## 👤 Author

Adedayo Adebayo
Data Analyst| ML Practitioner
