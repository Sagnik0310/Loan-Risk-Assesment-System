from database.fetch_data import FetchData
from structure.preprocessing import DataPreprocessor
from structure.train_models import TrainModels
from structure.hyperparameter_tuning import HyperparameterTuning
from structure.stacking_classifier import StackingEnsemble
from structure.evaluation import ModelEvaluation

from sklearn.model_selection import train_test_split


def main():

    print("=" * 80)
    print("LOAN RISK ASSESSMENT PIPELINE")
    print("=" * 80)

    # ======================================================
    # Fetch Data
    # ======================================================
    fetcher = FetchData()
    df = fetcher.get_dataframe()

    # ======================================================
    # Preprocess Data
    # ======================================================
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)

    # ======================================================
    # Split Dataset
    # ======================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ======================================================
    # Train Base Models
    # ======================================================
    trainer = TrainModels()
    trainer.train_models(X_train, y_train)

    # ======================================================
    # Hyperparameter Tuning
    # ======================================================
    tuner = HyperparameterTuning()
    tuner.tune_models(X_train, y_train)

    # ======================================================
    # Stacking Classifier
    # ======================================================
    stacking = StackingEnsemble()
    stacking.load_base_models()
    stacking.build_model()
    stacking.train(X_train, y_train)
    stacking.save_model()

    # ======================================================
    # Evaluation
    # ======================================================
    evaluator = ModelEvaluation()
    evaluator.load_models()
    evaluator.evaluate_models(X_test, y_test)

    print(evaluator.get_results_dataframe())

    evaluator.save_results()
    evaluator.save_classification_reports()
    evaluator.plot_roc_curves(X_test, y_test)
    evaluator.plot_precision_recall_curves(X_test, y_test)
    evaluator.print_cross_validation(
        evaluator.cross_validation(X, y)
    )

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()