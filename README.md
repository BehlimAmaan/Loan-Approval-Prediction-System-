# 🏦 Loan Approval Prediction System

An end-to-end **Machine Learning project** that predicts whether a loan will be **Approved or Rejected** based on applicant financial and personal details.
The system is trained using **Logistic Regression** and deployed as an **interactive Streamlit web application**.

---

## 📌 Project Overview

Banks and financial institutions need to assess loan applications efficiently while minimizing risk.
This project uses historical loan data to build a predictive model that helps in **automated loan eligibility decisions**.

The application allows users to enter applicant details and instantly get a **loan approval decision**.

---

## 🎯 Objective

* Predict loan approval status (`Approved / Rejected`)
* Reduce manual decision-making
* Demonstrate a complete ML pipeline with deployment

---

## 🧠 Machine Learning Details

* **Algorithm Used:** Logistic Regression
* **Problem Type:** Binary Classification
* **Target Variable:** `loan_status`

  * `1` → Approved
  * `0` → Rejected

### Key Techniques:

* Data cleaning (handling extra spaces, encoding categories)
* Feature scaling using `StandardScaler`
* Handling class imbalance using `class_weight='balanced'`
* Model evaluation using accuracy, confusion matrix, precision, recall, and F1-score

---

## 📊 Dataset Features

* Number of dependents
* Education
* Self-employed status
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential assets value
* Commercial assets value
* Luxury assets value
* Bank asset value

---

## 🧪 Model Performance

* **Accuracy:** ~78%
* Balanced precision and recall for both approved and rejected loans
* Reduced bias after handling class imbalance

---

## 🌐 Web Application (Streamlit)

The trained model is deployed using **Streamlit**, providing:

* User-friendly interface
* Real-time predictions
* Clean, modern dashboard-style UI

---

## 🚀 Deployment

The application is deployed online using **Streamlit Community Cloud**.

Users can:

* Enter applicant details
* Click “Predict Loan Status”
* Instantly see the decision

---

## 🗂️ Project Structure

```
Loan-Approval-Prediction/
│
├── app.py
├── requirements.txt
│
├── Train Model/
│   ├── EDA.ipynb
│   ├── Train.ipynb
│   ├── #Dataset
|
├── Model/
│   ├── loan_prediction_model.pkl
│   ├── scaler.pkl
│
└── README.md

---

## ⚙️ Installation & Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧾 Requirements

```
streamlit
scikit-learn
numpy
joblib
```

---

## 🧠 Skills Demonstrated

* Machine Learning (Classification)
* Data Preprocessing & Feature Engineering
* Model Evaluation
* Handling Class Imbalance
* Model Serialization
* Streamlit Deployment
* End-to-End ML Workflow

---

## 📌 Future Improvements

* Add prediction probability (%)
* Add risk level (Low / Medium / High)
* Try advanced models (Random Forest, XGBoost)
* Improve UI with charts and insights
* Add authentication for users

---

## 👨‍💻 Author

**Amaan Behlim**
CSE (AI/ML) Student
Aspiring ML Engineer

---

## ⭐ Final Note

This project demonstrates a **complete ML lifecycle** — from data preprocessing and model training to deployment and real-world usage.

If you like this project, feel free to ⭐ the repository!


