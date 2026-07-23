"""
hyperparameter_tuning.py

Purpose
-------
Tune all base models using RandomizedSearchCV and save the
best estimators for stacking.
"""

import os
import joblib
import pandas as pd

from scipy.stats import randint
from scipy.stats import uniform

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier


class HyperparameterTuning:

    def __init__(self):

        self.cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        self.models = {

            "Logistic Regression": (
                LogisticRegression(random_state=42, max_iter=1000),
                {
                    "C": uniform(0.01, 10),
                    "penalty": ["l2"],
                    "solver": ["lbfgs"]
                }
            ),

            "Support Vector Machine": (
                SVC(probability=True, random_state=42),
                {
                    "C": uniform(0.1, 20),
                    "kernel": ["rbf", "poly"],
                    "gamma": ["scale", "auto"]
                }
            ),

            "K Nearest Neighbors": (
                KNeighborsClassifier(),
                {
                    "n_neighbors": randint(3, 15),
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                }
            ),

            "Decision Tree": (
                DecisionTreeClassifier(random_state=42),
                {
                    "criterion": ["gini", "entropy"],
                    "max_depth": randint(3, 20),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10)
                }
            ),

            "Random Forest": (
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": randint(100, 500),
                    "max_depth": randint(5, 25),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10),
                    "max_features": ["sqrt", "log2"]
                }
            ),

            "Naive Bayes": (
                GaussianNB(),
                {
                    "var_smoothing": uniform(1e-10, 1e-8)
                }
            ),

            "XGBoost": (
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                ),
                {
                    "n_estimators": randint(100, 500),
                    "learning_rate": uniform(0.01, 0.2),
                    "max_depth": randint(3, 10),
                    "min_child_weight": randint(1, 10),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                    "gamma": uniform(0, 0.5)
                }
            )

        }

    def tune_models(self, X_train, y_train):

        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        tuned_models = {}

        tuning_results = []

        print("=" * 70)
        print("Hyperparameter Tuning Started")
        print("=" * 70)

        for name, (model, params) in self.models.items():

            print(f"\nTuning {name}...")

            random_search = RandomizedSearchCV(

                estimator=model,

                param_distributions=params,

                n_iter=25,

                scoring="f1",

                cv=self.cv,

                verbose=1,

                random_state=42,

                n_jobs=-1

            )

            random_search.fit(

                X_train,

                y_train

            )

            best_model = random_search.best_estimator_

            tuned_models[name] = best_model

            filename = (
                name.lower()
                .replace(" ", "_")
                + "_tuned.pkl"
            )

            joblib.dump(
                best_model,
                os.path.join("models", filename)
            )

            tuning_results.append({

                "Model": name,

                "Best Score": random_search.best_score_,

                "Best Parameters": random_search.best_params_

            })

            print(f"Finished tuning {name}")

        results = pd.DataFrame(tuning_results)

        results.to_csv(

            "reports/hyperparameter_results.csv",

            index=False

        )

        print("\nHyperparameter tuning completed.")

        return tuned_models, results