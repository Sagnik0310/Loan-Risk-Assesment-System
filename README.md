# 💳 LoanPredict — Enterprise Loan Risk Assessment System
### Intelligent Credit Risk Analysis using Ensemble Learning & Stacking Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/MongoDB-Database-darkgreen?style=for-the-badge&logo=mongodb">
  <img src="https://img.shields.io/badge/Streamlit-Fintech%20Dashboard-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Analytics-3F4F75?style=for-the-badge&logo=plotly">
</p>

---

## 📖 Project Overview

Financial institutions evaluate thousands of loan applications daily. Approving high-risk loans without rigorous quantitative analysis increases default exposure, while rejecting creditworthy applicants results in lost lending opportunities.

**LoanPredict (Loan Risk Assessment System)** is an end-to-end Machine Learning credit underwriting application that estimates whether a loan application represents a **Low Risk (`Class 0`)** or **High Risk (`Class 1`, `not.fully.paid = 1`)** profile by analyzing 13 financial, underwriting, and credit-bureau attributes.

Instead of relying on a single classifier, LoanPredict employs a two-level **Stacking Ensemble** (`StackingClassifier`) combining seven independently tuned supervised learning algorithms with a **Logistic Regression** meta-learner. The platform integrates **MongoDB** for persistent application history and audit logging, and features a modern, institutional **Streamlit + Plotly** fintech interface.

---

## ✨ Key Features

- **End-to-End Machine Learning Pipeline**: Automated preprocessing (`StandardScaler`, `LabelEncoder`), base model training, `RandomizedSearchCV` hyperparameter tuning, and ensemble stacking.
- **5-Module Institutional Fintech UI**:
  1. **Overview Dashboard**: Real-time portfolio KPIs, risk distribution donut chart, loan-purpose stacked breakdown, financial attribute summaries, and recent assessment activity with privacy-masked contact fields.
  2. **New Loan Assessment**: Guided 4-stage underwriting workstation (Applicant Profile, Loan Structure & Policy, Income & Debt Capacity, Credit Bureau Indicators), sample profile presets, duplicate-submission protection, and an interactive Plotly risk probability gauge.
  3. **Application History**: Multi-criteria filtering (text search, risk category, loan purpose, date range), dynamic sorting, pagination, CSV export, PII masking toggle, exact backend MongoDB lookup (`Name + Email + Phone`), full application inspector, and confirmation-guarded record deletion.
  4. **Analytics Dashboard**: Interactive Plotly visualizations (assessment volume over time, FICO box distributions by risk class, interest rate vs. default probability scatter plot, purpose breakdown) alongside portfolio financial statistics and credit risk flags.
  5. **Model Insights**: Read-only benchmark explorer displaying offline test-set metrics (`reports/model_comparison.csv`), 5-fold stratified cross-validation results (`reports/cross_validation_results.csv`), tuned hyperparameters, per-model JSON classification reports, and saved ROC & Precision-Recall curve artifacts.
- **Adaptive Dark & Light Fintech Themes**: Seamless theme switching with institutional navy, slate, and teal styling.
- **MongoDB Persistence & Audit Trail**: Real-time storage and retrieval of applicant metadata, 13 model features, predicted risk class, and class-1 probability.

---

## 🏗️ Project Architecture

```text
                        Loan Dataset (MongoDB)
                                  │
                                  ▼
                      Data Retrieval (FetchData)
                                  │
                                  ▼
           Data Preprocessing (LabelEncoder + StandardScaler)
                                  │
                                  ▼
               7 Base Classifiers + RandomizedSearchCV
                                  │
                                  ▼
         Ensemble Learning (StackingClassifier + Meta-Learner)
                                  │
                                  ▼
            Saved Model Artifacts (models/stacking_classifier.pkl)
                                  │
                                  ▼
          Streamlit Fintech Platform (app.py + ui/* modules)
                                  │
                                  ▼
        Real-Time Risk Assessment ◄──► MongoDB Prediction History
```

---

## 📂 Project Structure

```text
Loan-Risk-Assesment-System/
│
├── .streamlit/
│   └── config.toml                  # Streamlit server & base theme configuration
│
├── database/
│   ├── db_collections.py            # MongoDB collection name definitions
│   ├── fetch_data.py                # Retrieves training dataset from MongoDB
│   ├── mongodb.py                   # MongoDB connection manager using .env
│   └── prediction_history.py        # CRUD operations for logged loan assessments
│
├── models/                          # Serialized encoders, scalers, base, tuned & stacking models
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── stacking_classifier.pkl
│   └── *_tuned.pkl
│
├── reports/                         # Offline evaluation CSVs, JSON reports & ROC/PR plots
│   ├── cross_validation_results.csv
│   ├── hyperparameter_results.csv
│   ├── model_comparison.csv
│   ├── *_classification_report.json
│   ├── precision_recall_curve.png
│   └── roc_curve.png
│
├── structure/                       # Core ML training, evaluation & inference modules
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── hyperparameter_tuning.py
│   ├── pipeline.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── stacking_classifier.py
│   └── train_models.py
│
├── ui/                              # Modular Fintech UI design system & page views
│   ├── __init__.py
│   ├── components.py                # Sidebar, headers, KPI cards, PII masking, chart themes
│   ├── pages.py                     # Overview, Assessment, History, Analytics & Model Insights views
│   ├── services.py                  # Cached model/report loaders & safe MongoDB wrappers
│   └── styles.py                    # Dark/Light fintech CSS design system
│
├── app.py                           # Main Streamlit web application entry point
├── requirements.txt                 # Python dependencies
├── Test_file.py                     # CLI utility for running single predictions
├── test_search.py                   # CLI utility for searching MongoDB records
└── test_ui_regression.py            # Automated ML regression, CRUD & Streamlit UI test suite
```

---

## 🧠 Machine Learning Pipeline & Stacking Ensemble

### Input Features (13 Model Attributes)
| Feature Name | Description | Type / Range |
| :--- | :--- | :--- |
| `credit.policy` | `1` if applicant meets underwriting credit criteria; `0` otherwise | Binary (`0` or `1`) |
| `purpose` | Purpose of the loan (`debt_consolidation`, `credit_card`, `home_improvement`, `major_purchase`, `small_business`, `educational`, `all_other`) | Categorical (`LabelEncoder`) |
| `int.rate` | Annual interest rate of the loan (decimal, e.g. `0.11` = `11%`) | Float (`0.0` – `1.0`) |
| `installment` | Monthly installment owed by the borrower if funded | Float (`$`) |
| `log.annual.inc` | Natural log of the self-reported annual income | Float |
| `dti` | Debt-to-income ratio of the borrower | Float (`%`) |
| `fico` | FICO credit score of the borrower | Integer (`300` – `850`) |
| `days.with.cr.line` | Number of days borrower has had a credit line | Float |
| `revol.bal` | Borrower's revolving credit balance | Float (`$`) |
| `revol.util` | Revolving line utilization rate (`%` of available credit used) | Float (`%`) |
| `inq.last.6mths` | Number of creditor inquiries in the last 6 months | Integer |
| `delinq.2yrs` | Number of times 30+ days past due in the last 2 years | Integer |
| `pub.rec` | Number of derogatory public records | Integer |

### Level-1 Base Estimators & Level-2 Meta-Learner
1. **Logistic Regression** (`logistic_regression_tuned.pkl`)
2. **Support Vector Machine (SVM)** (`support_vector_machine_tuned.pkl`)
3. **K-Nearest Neighbors (KNN)** (`k_nearest_neighbors_tuned.pkl`)
4. **Decision Tree** (`decision_tree_tuned.pkl`)
5. **Random Forest** (`random_forest_tuned.pkl`)
6. **Gaussian Naive Bayes** (`naive_bayes_tuned.pkl`)
7. **XGBoost** (`xgboost_tuned.pkl`)
8. **Meta-Learner**: **Logistic Regression** combining 5-fold cross-validated `predict_proba` outputs (`models/stacking_classifier.pkl`).

---

## ⚙️ Setup, Configuration & Running the Application

### 1. Environment Configuration
Ensure a `.env` file exists in the project root (`Loan-Risk-Assesment-System/.env`) with your MongoDB connection settings:
```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/
DATABASE_NAME=<your_database_name>
COLLECTION_NAME=<your_training_collection_name>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
cd Loan-Risk-Assesment-System
streamlit run app.py
```

### 4. Run Automated Regression & UI Tests
```bash
cd Loan-Risk-Assesment-System
python test_ui_regression.py
```

---

## 👨‍💻 Authors

Founded & Developed by **Sagnik Ghosh & Anantika Ghosh** · Version 1.0.0
