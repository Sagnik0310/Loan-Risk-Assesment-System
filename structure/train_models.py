"""
train_models.py

Purpose:
--------
Train all base machine learning models for the Loan Risk Assessment System.

Models:
1. Logistic Regression
2. Support Vector Machine
3. K-Nearest Neighbors
4. Decision Tree
5. Random Forest
6. Gaussian Naive Bayes
7. XGBoost
"""

import os
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier


class TrainModels:

    def __init__(self):

        """
        Initialize all baseline models.

        NOTE:
        These are strong baseline hyperparameters.
        They will be optimized later using RandomizedSearchCV.
        """

        self.models = {

            "Logistic Regression": LogisticRegression(
                random_state=42,
                max_iter=1000
            ),

            "Support Vector Machine": SVC(
                probability=True,
                random_state=42
            ),

            "K Nearest Neighbors": KNeighborsClassifier(),

            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),

            "Naive Bayes": GaussianNB(),

            "XGBoost": XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            )

        }

    def train_models(self, X, y):

        """
        Train all base models.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix

        y : pandas.Series
            Target column

        Returns
        -------
        trained_models
        X_train
        X_test
        y_train
        y_test
        """

        print("=" * 70)
        print("Splitting Dataset")
        print("=" * 70)

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            stratify=y,

            random_state=42

        )

        os.makedirs("models", exist_ok=True)

        trained_models = {}

        print("\n")

        print("=" * 70)
        print("Training Base Models")
        print("=" * 70)

        for model_name, model in self.models.items():

            print(f"\nTraining {model_name}...")

            model.fit(

                X_train,

                y_train

            )

            trained_models[model_name] = model

            filename = (

                model_name

                .lower()

                .replace(" ", "_")

                + ".pkl"

            )

            joblib.dump(

                model,

                os.path.join(

                    "models",

                    filename

                )

            )

            print(f"{model_name} saved successfully.")

        print("\n")

        print("=" * 70)
        print("All Base Models Trained Successfully")
        print("=" * 70)

        return (

            trained_models,

            X_train,

            X_test,

            y_train,

            y_test

        )


if __name__ == "__main__":

    print(
        "Import this class into pipeline.py"
    )



