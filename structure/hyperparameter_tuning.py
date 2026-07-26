"""
hyperparameter_tuning.py

Purpose
-------
Tune all base models using RandomizedSearchCV and save the
best estimators.
"""

import os
import time
import joblib
import warnings
import pandas as pd

from scipy.stats import randint
from scipy.stats import uniform

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
class HyperparameterTuning:
    def __init__(self):

        self.cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

        self.models = {

        "Logistic Regression": (

            LogisticRegression(
                random_state=42,
                max_iter=1000
            ),

            {
                "C": uniform(0.01, 10),
                "solver": ["lbfgs"]
            }

        ),

        "Support Vector Machine": (

            SVC(
                random_state=42,
                probability=False
            ),

            {
                "C": uniform(0.1, 5),
                "kernel": ["rbf"],
                "gamma": ["scale"]
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

            DecisionTreeClassifier(
                random_state=42
            ),

            {
                "criterion": ["gini", "entropy"],
                "max_depth": randint(3, 20),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10)
            }

        ),

        "Random Forest": (

            RandomForestClassifier(
                random_state=42
            ),

            {
                "n_estimators": randint(50, 150),
                "max_depth": randint(5, 20),
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 5),
                "max_features": ["sqrt"]
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
                "n_estimators": randint(50, 150),
                "learning_rate": uniform(0.01, 0.2),
                "max_depth": randint(3, 8),
                "subsample": uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.7, 0.3)
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

            print("\n" + "=" * 70)
            print(f"Tuning {name}")
            print("=" * 70)

            start_time = time.time()

            search = RandomizedSearchCV(

                estimator=model,

                param_distributions=params,

                n_iter=5,

                scoring="f1",

                cv=self.cv,

                random_state=42,

                n_jobs=-1,

                verbose=2,

                error_score="raise"

            )

            try:

                search.fit(X_train, y_train)

                best_model = search.best_estimator_

                tuned_models[name] = best_model

                filename = (
                    name.lower()
                    .replace(" ", "_")
                    + "_tuned.pkl"
                )

                filepath = os.path.join(
                    "models",
                    filename
                )

                joblib.dump(
                    best_model,
                    filepath
                )
                tuning_results.append({

                    "Model": name,

                    "Best Score": round(
                        search.best_score_,
                        4
                    ),

                    "Best Parameters": search.best_params_

                })

                elapsed = time.time() - start_time

                print("\n✓ Model Tuned Successfully")
                print(f"Model      : {name}")
                print(f"Best Score : {search.best_score_:.4f}")
                print(f"Time Taken : {elapsed:.2f} seconds")
                print(f"Saved File : {filename}")

            except Exception as e:

                print(f"\n❌ Error while tuning {name}")
                print(e)

        results = pd.DataFrame(tuning_results)

        results.to_csv(

            "reports/hyperparameter_results.csv",

            index=False

        )

        print("\n" + "=" * 70)
        print("Hyperparameter Tuning Completed")
        print("=" * 70)

        print(results)

        return tuned_models, results