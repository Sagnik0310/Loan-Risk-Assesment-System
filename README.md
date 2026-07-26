# 💳 Loan Risk Assessment System
### Intelligent Loan Approval Prediction using Ensemble Learning and Stacking Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/MongoDB-Database-darkgreen?style=for-the-badge&logo=mongodb">
  <img src="https://img.shields.io/badge/Streamlit-Web%20Application-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/CSS-Frontend-blue?style=for-the-badge&logo=css3">
</p>

---

# 📖 Project Overview

Financial institutions receive thousands of loan applications every day. Approving loans without evaluating an applicant's financial profile can lead to a significant increase in loan defaults, whereas rejecting genuine applicants can result in missed business opportunities.

The **Loan Risk Assessment System** is an end-to-end Machine Learning application that predicts whether a loan application should be approved or rejected by analyzing multiple financial and personal attributes of the applicant.

Instead of relying on a single machine learning algorithm, this project adopts an **Ensemble Learning** approach where several classification algorithms are trained independently. Their predictions are then combined using a **Stacking Classifier**, allowing the system to make more accurate, reliable, and generalized predictions.

The project also integrates **MongoDB** for efficient data storage, uses **Scikit-Learn** and **XGBoost** for model development, and is deployed through an interactive **Streamlit** web application enhanced with custom **CSS** styling.

---

# 🎯 Problem Statement

Banks and financial organizations need an intelligent system capable of determining whether a customer is likely to repay a loan based on historical data.

The objective of this project is to build a machine learning model capable of predicting loan approval by considering various applicant characteristics such as income, employment details, credit history, loan amount, education, marital status, existing debts, and several other financial indicators.

The final prediction assists financial institutions in making informed lending decisions while minimizing the overall credit risk.

---

# ✨ Features

- Complete End-to-End Machine Learning Pipeline
- MongoDB Database Integration
- Automated Data Preprocessing
- Feature Engineering
- Hyperparameter Tuning
- Training of Multiple Machine Learning Models
- Ensemble Learning using Stacking Classifier
- Model Performance Evaluation
- Interactive Streamlit Web Application
- Responsive Frontend with Custom CSS
- Real-Time Loan Prediction

---

# 🏗️ Project Architecture

```
                        Loan Dataset
                             │
                             ▼
                     MongoDB Database
                             │
                             ▼
                    Data Retrieval Module
                             │
                             ▼
                     Data Preprocessing
                             │
                             ▼
                    Feature Engineering
                             │
                             ▼
                  Individual Model Training
                             │
                             ▼
                  Hyperparameter Tuning
                             │
                             ▼
                 Ensemble Learning (Stacking)
                             │
                             ▼
                   Final Trained Model
                             │
                             ▼
                  Streamlit Application
                             │
                             ▼
                  Loan Approval Prediction
```

---

# 📂 Project Structure

```
## 📂 Project Structure

```text
Loan-Risk-Assessment-System/
│
├── __pycache__/
│   └── Test_file.cpython-314.pyc
│
├── .streamlit/
│   └── config.toml
│
├── database/
│   ├── __pycache__/
│   ├── db_collections.py
│   ├── fetch_data.py
│   ├── mongodb.py
│   └── prediction_history.py
│
├── models/
│
├── reports/
│
├── structure/
│   ├── __pycache__/
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── hyperparameter_tuning.py
│   ├── pipeline.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── stacking_classifier.py
│   └── train_models.py
│
├── .gitattributes
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
├── Test_file.py
└── test_search.py
```
```

---

# 🗄️ MongoDB Integration

MongoDB serves as the primary database of the project.

The raw loan dataset is first imported into MongoDB. Instead of repeatedly loading CSV files during development, the application retrieves the data directly from the MongoDB collection, making the workflow more scalable and closer to real-world production systems.

MongoDB stores:

- Applicant Information
- Loan Details
- Financial Attributes
- Historical Records
- Prediction Inputs

### Workflow

```
CSV Dataset
      │
      ▼
MongoDB Collection
      │
      ▼
Fetch Data using PyMongo
      │
      ▼
Pandas DataFrame
      │
      ▼
Machine Learning Pipeline
```

---

# ⚙️ Data Preprocessing

Machine Learning models require clean and properly formatted data to produce reliable predictions. Before training the models, several preprocessing steps are performed.

The preprocessing pipeline includes:

- Removing duplicate records
- Handling missing values
- Dropping unnecessary columns
- Encoding categorical variables using Label Encoding
- Feature Scaling using StandardScaler
- Splitting the dataset into Training and Testing sets

Proper preprocessing ensures that every model receives standardized numerical input, improving both training efficiency and prediction accuracy.

---

# 🧠 Machine Learning Models

Instead of relying on a single algorithm, this project trains multiple supervised learning models independently. Each model has different strengths and learns different characteristics of the dataset.

The predictions from all these models are later combined through a Stacking Classifier.

---

# 1️⃣ Logistic Regression

Logistic Regression is a supervised classification algorithm used for binary prediction problems. It estimates the probability that a given applicant belongs to either the approved or rejected loan category using the sigmoid function.

Despite being one of the simplest classification algorithms, Logistic Regression performs remarkably well on structured financial datasets where relationships between features are approximately linear.

In this project, Logistic Regression acts as one of the **base learners**, providing probability-based predictions that contribute to the final ensemble model.

### Advantages

- Fast training
- Computationally efficient
- Produces probability scores
- Easy to interpret
- Strong baseline classifier

---

# 2️⃣ Support Vector Machine (SVM)

Support Vector Machine is a powerful supervised learning algorithm designed to find the optimal decision boundary between different classes.

Instead of simply separating the classes, SVM attempts to maximize the margin between them, resulting in improved generalization on unseen data.

For loan prediction, SVM is capable of identifying complex relationships between applicant attributes that may not be captured by simpler linear models.

### Advantages

- Excellent classification accuracy
- Handles high-dimensional data
- Effective with nonlinear decision boundaries
- Robust against overfitting

---

# 3️⃣ Decision Tree

Decision Trees classify data by recursively splitting the dataset according to the most informative feature.

Each internal node represents a decision based on one feature, while each leaf node represents the final class prediction.

Decision Trees are highly interpretable because every prediction follows a clear sequence of decision rules.

In this project, they capture nonlinear relationships between financial variables that linear models may fail to identify.

### Advantages

- Easy to understand
- No feature scaling required
- Captures nonlinear patterns
- Handles categorical features effectively

---

# 4️⃣ Random Forest

Random Forest is an ensemble learning algorithm based on multiple Decision Trees.

Each tree is trained using:

- Random subsets of training samples (Bagging)
- Random subsets of features

Instead of depending on one Decision Tree, Random Forest aggregates predictions from hundreds of trees through majority voting, reducing overfitting and improving prediction stability.

### Advantages

- High prediction accuracy
- Robust against overfitting
- Handles missing information well
- Performs well on structured datasets

---

# 5️⃣ K-Nearest Neighbors (KNN)

KNN is an instance-based learning algorithm.

Instead of learning explicit mathematical relationships, it predicts the class of a new applicant by examining the nearest training samples.

The assumption is that applicants with similar financial characteristics are likely to have similar loan outcomes.

### Advantages

- Simple implementation
- No training phase
- Effective for smaller datasets
- Captures local patterns

---

# 6️⃣ Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' Theorem.

It assumes that each feature contributes independently to the prediction and follows a Gaussian (normal) distribution.

Although the independence assumption rarely holds perfectly in real-world financial data, Naive Bayes often provides surprisingly competitive results while being extremely fast.

### Advantages

- Extremely fast
- Works well on small datasets
- Handles probabilistic predictions efficiently
- Requires minimal computational resources

---

# 7️⃣ XGBoost

XGBoost (Extreme Gradient Boosting) is one of the most powerful ensemble learning algorithms used in modern machine learning.

Unlike Random Forest, which builds trees independently, XGBoost builds trees sequentially. Each new tree focuses on correcting the mistakes made by previous trees, gradually improving overall model performance.

Due to its ability to capture complex nonlinear relationships, XGBoost is widely used in finance, banking, healthcare, and many Kaggle competitions.

### Advantages

- High predictive accuracy
- Regularization reduces overfitting
- Handles missing values
- Fast and scalable
- State-of-the-art performance on structured datasets

---

# 🔍 Hyperparameter Tuning

Training a model using default parameters rarely produces the best performance.

Therefore, this project applies **GridSearchCV** to systematically search for the optimal combination of hyperparameters for each machine learning algorithm.

Some of the tuned parameters include:

- Number of Estimators
- Maximum Depth
- Learning Rate
- Kernel Type
- Regularization Parameter
- Number of Neighbors

Hyperparameter tuning significantly improves model accuracy and ensures better generalization on unseen loan applications.

---

# 🤝 Ensemble Learning

Every machine learning algorithm has its own strengths and weaknesses.

Some models may perform exceptionally well on one type of applicant but poorly on another. Instead of selecting a single "best" model, Ensemble Learning combines the knowledge of multiple models to produce a stronger and more reliable prediction.

The underlying principle of Ensemble Learning is that a collection of diverse models often outperforms any individual model.

Benefits include:

- Higher prediction accuracy
- Better generalization
- Lower variance
- Reduced overfitting
- Improved robustness

---

# 🏆 Stacking Classifier

The primary ensemble technique used in this project is **Stacking (Stacked Generalization)**.

Unlike traditional ensemble methods such as Bagging or Boosting, Stacking combines multiple machine learning models by introducing another model known as the **Meta Learner**.

The workflow consists of two levels:

### Level 1 - Base Models

The following algorithms are trained independently using the same training dataset:

- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Gaussian Naive Bayes
- XGBoost

Each of these models generates its own prediction for every applicant.

```
Training Dataset

        │
        ▼

 ┌──────────────┐
 │ Logistic Reg │
 └──────────────┘

 ┌──────────────┐
 │     SVM      │
 └──────────────┘

 ┌──────────────┐
 │Decision Tree │
 └──────────────┘

 ┌──────────────┐
 │Random Forest │
 └──────────────┘

 ┌──────────────┐
 │     KNN      │
 └──────────────┘

 ┌──────────────┐
 │ Naive Bayes  │
 └──────────────┘

 ┌──────────────┐
 │   XGBoost    │
 └──────────────┘
```

Instead of selecting one prediction, every model contributes its output.

---

### Level 2 - Meta Learner

The predictions generated by all the base models become the input features for another machine learning model called the **Meta Learner**.

```
Logistic Regression Prediction
              │
SVM Prediction
              │
Decision Tree Prediction
              │
Random Forest Prediction
              │
KNN Prediction
              │
Naive Bayes Prediction
              │
XGBoost Prediction
              │
              ▼
      Meta Learner
              │
              ▼
 Final Loan Approval Prediction
```

The Meta Learner learns which base model performs best under different situations and intelligently combines their outputs to produce the final prediction.

This two-level learning strategy generally achieves higher accuracy than relying on a single machine learning model.

---

# 📊 Model Evaluation

The trained models are evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Comparing these metrics enables the selection of the most reliable ensemble model for deployment.

---

# 🌐 Streamlit Web Application

The trained stacking model is deployed using **Streamlit** through the `app.py` file.

The web application provides a simple and interactive interface where users can enter applicant details and instantly receive a loan prediction.

The application includes:

- User-friendly input forms
- Real-time prediction
- Clean and responsive interface
- Fast model inference
- Easy deployment

---

# 🎨 Frontend

The frontend of the application is developed using:

- Streamlit
- Custom CSS (`style.css`)

The CSS file enhances the overall appearance of the application by improving:

- Layout
- Typography
- Buttons
- Input Components
- Cards
- Colors
- Responsiveness

This provides users with a clean, modern, and intuitive interface for interacting with the prediction system.

---

# 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-Learn, XGBoost |
| Database | MongoDB |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Frontend | Streamlit, CSS |
| Visualization | Matplotlib, Seaborn |

---

# 🚀 Future Enhancements

- Explainable AI using SHAP and LIME
- Cloud Deployment (AWS/Azure/GCP)
- REST API Integration
- Docker Containerization
- Continuous Model Retraining
- User Authentication
- Loan Analytics Dashboard
- Automated Data Validation

---

# 👨‍💻 Conclusion

The **Loan Risk Assessment System** demonstrates the complete lifecycle of a real-world Machine Learning project, from data storage and preprocessing to model development, evaluation, ensemble learning, and deployment.

By combining multiple classification algorithms through a **Stacking Classifier**, the system leverages the strengths of each individual model to achieve more accurate and robust loan approval predictions. The integration of **MongoDB** ensures efficient data management, while the **Streamlit** application with custom **CSS** provides a clean and interactive user experience, making the solution both technically comprehensive and practically applicable.