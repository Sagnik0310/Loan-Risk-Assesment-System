import os
import joblib

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


class StackingEnsemble:

    def __init__(self):

        self.base_models = {}

        self.stacking_model = None

    def load_base_models(self):

        model_files = {

            "lr": "models/logistic_regression_tuned.pkl",

            "svm": "models/support_vector_machine_tuned.pkl",

            "knn": "models/k_nearest_neighbors_tuned.pkl",

            "dt": "models/decision_tree_tuned.pkl",

            "rf": "models/random_forest_tuned.pkl",

            "nb": "models/naive_bayes_tuned.pkl",

            "xgb": "models/xgboost_tuned.pkl"

        }

        for name, path in model_files.items():

            self.base_models[name] = joblib.load(path)

    def build_model(self):

        estimators = [

            ("lr", self.base_models["lr"]),

            ("svm", self.base_models["svm"]),

            ("knn", self.base_models["knn"]),

            ("dt", self.base_models["dt"]),

            ("rf", self.base_models["rf"]),

            ("nb", self.base_models["nb"]),

            ("xgb", self.base_models["xgb"])

        ]

        self.stacking_model = StackingClassifier(

            estimators=estimators,

            final_estimator=LogisticRegression(

                random_state=42,

                max_iter=1000

            ),

            stack_method="predict_proba",

            cv=5,

            n_jobs=-1,

            passthrough=False

        )

    def train(self, X_train, y_train):

        self.stacking_model.fit(

            X_train,

            y_train

        )

    def save_model(self):

        os.makedirs(

            "models",

            exist_ok=True

        )

        joblib.dump(

            self.stacking_model,

            "models/stacking_classifier.pkl"

        )

    def get_model(self):

        return self.stacking_model