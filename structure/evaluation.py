import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)


class ModelEvaluation:

    def __init__(self):

        self.models = {}
        self.results = []
        self.classification_reports = {}
        self.confusion_matrices = {}

    ####################################################################
    # Load all trained models
    ####################################################################

    def load_models(self):

        model_files = {

            "Logistic Regression":
                "models/logistic_regression_tuned.pkl",

            "Support Vector Machine":
                "models/support_vector_machine_tuned.pkl",

            "K Nearest Neighbors":
                "models/k_nearest_neighbors_tuned.pkl",

            "Decision Tree":
                "models/decision_tree_tuned.pkl",

            "Random Forest":
                "models/random_forest_tuned.pkl",

            "Naive Bayes":
                "models/naive_bayes_tuned.pkl",

            "XGBoost":
                "models/xgboost_tuned.pkl"
        }

        for model_name, path in model_files.items():

            if os.path.exists(path):

                self.models[model_name] = joblib.load(path)

                print(f"{model_name} Loaded Successfully")

            else:

                print(f"{path} Not Found")

    ####################################################################
    # Evaluate Models
    ####################################################################

    def evaluate_models(self, X_test, y_test):

        self.results = []

        for name, model in self.models.items():

            y_pred = model.predict(X_test)

            y_prob = None

            if hasattr(model, "predict_proba"):

                y_prob = model.predict_proba(X_test)[:, 1]

            elif hasattr(model, "decision_function"):

                y_prob = model.decision_function(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            if y_prob is not None:

                roc_auc = roc_auc_score(
                    y_test,
                    y_prob
                )

                loss = log_loss(
                    y_test,
                    y_prob
                )

            else:

                roc_auc = np.nan
                loss = np.nan

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            matrix = confusion_matrix(
                y_test,
                y_pred
            )

            self.classification_reports[name] = report
            self.confusion_matrices[name] = matrix

            self.results.append({

                "Model": name,

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1 Score": f1,

                "ROC AUC": roc_auc,

                "Log Loss": loss

            })

    ####################################################################
    # Results DataFrame
    ####################################################################

    def get_results_dataframe(self):

        results_df = pd.DataFrame(self.results)

        results_df = results_df.sort_values(

            by="ROC AUC",

            ascending=False

        )

        return results_df

    ####################################################################
    # Save Results
    ####################################################################

    def save_results(self):

        os.makedirs(

            "reports",

            exist_ok=True

        )

        results_df = self.get_results_dataframe()

        results_df.to_csv(

            "reports/model_comparison.csv",

            index=False

        )

        print("Model comparison saved.")

    ####################################################################
    # Save Classification Reports
    ####################################################################

    def save_classification_reports(self):

        os.makedirs(

            "reports",

            exist_ok=True

        )

        for name, report in self.classification_reports.items():

            filename = name.lower().replace(" ", "_")

            with open(

                f"reports/{filename}_classification_report.json",

                "w"

            ) as file:

                json.dump(

                    report,

                    file,

                    indent=4

                )

    ####################################################################
    # Plot ROC Curves
    ####################################################################

    def plot_roc_curves(self, X_test, y_test):

        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():

            y_prob = None

            if hasattr(model, "predict_proba"):

                y_prob = model.predict_proba(X_test)[:, 1]

            elif hasattr(model, "decision_function"):

                y_prob = model.decision_function(X_test)

            if y_prob is None:

                continue

            fpr, tpr, _ = roc_curve(

                y_test,

                y_prob

            )

            roc_auc = auc(

                fpr,

                tpr

            )

            plt.plot(

                fpr,

                tpr,

                linewidth=2,

                label=f"{name} (AUC={roc_auc:.3f})"

            )

        plt.plot(

            [0, 1],

            [0, 1],

            linestyle="--"

        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curves")

        plt.legend()

        os.makedirs(

            "reports",

            exist_ok=True

        )

        plt.savefig(

            "reports/roc_curve.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    ####################################################################
    # Plot Precision Recall Curves
    ####################################################################

    def plot_precision_recall_curves(self, X_test, y_test):

        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():

            y_prob = None

            if hasattr(model, "predict_proba"):

                y_prob = model.predict_proba(X_test)[:, 1]

            elif hasattr(model, "decision_function"):

                y_prob = model.decision_function(X_test)

            if y_prob is None:

                continue

            precision, recall, _ = precision_recall_curve(

                y_test,

                y_prob

            )

            pr_auc = auc(

                recall,

                precision

            )

            plt.plot(

                recall,

                precision,

                linewidth=2,

                label=f"{name} (AUC={pr_auc:.3f})"

            )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision Recall Curves")

        plt.legend()

        os.makedirs(

            "reports",

            exist_ok=True

        )

        plt.savefig(

            "reports/precision_recall_curve.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

    ####################################################################
    # Cross Validation
    ####################################################################

    def cross_validation(self, X, y):

        cv = StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=42

        )

        cv_results = []

        for name, model in self.models.items():

            scores = cross_val_score(

                model,

                X,

                y,

                cv=cv,

                scoring="roc_auc",

                n_jobs=-1

            )

            cv_results.append({

                "Model": name,

                "Mean ROC AUC": scores.mean(),

                "Standard Deviation": scores.std(),

                "Fold 1": scores[0],

                "Fold 2": scores[1],

                "Fold 3": scores[2],

                "Fold 4": scores[3],

                "Fold 5": scores[4]

            })

        cv_df = pd.DataFrame(cv_results)

        cv_df = cv_df.sort_values(

            by="Mean ROC AUC",

            ascending=False

        )

        os.makedirs(

            "reports",

            exist_ok=True

        )

        cv_df.to_csv(

            "reports/cross_validation_results.csv",

            index=False

        )

        return cv_df

    ####################################################################
    # Print Cross Validation Results
    ####################################################################

    def print_cross_validation(self, cv_df):

        print()

        print("=" * 80)

        print("Cross Validation Summary")

        print("=" * 80)

        print(cv_df)

        print("=" * 80)