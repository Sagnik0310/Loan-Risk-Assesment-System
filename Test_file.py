from database.fetch_data import FetchData
from structure.preprocessing import DataPreprocessor
from structure.train_models import TrainModels
from structure.hyperparameter_tuning import HyperparameterTuning

from sklearn.model_selection import train_test_split


def main():

    # Fetch Dataset
    fetcher = FetchData()
    df = fetcher.get_dataframe()

    # Preprocess Dataset
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)

    print("=" * 70)
    print("Splitting Dataset")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train Base Models
    trainer = TrainModels()
    trainer.train_models(X_train, y_train)

    # Hyperparameter Tuning
    tuner = HyperparameterTuning()

    tuned_models, results = tuner.tune_models(
        X_train,
        y_train
    )

    print("\n")
    print("=" * 70)
    print("Final Results")
    print("=" * 70)

    print(results)


if __name__ == "__main__":
    main()