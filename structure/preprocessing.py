import os   #file handling
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:

    def __init__(self):

        self.scaler = StandardScaler()

        self.encoder = LabelEncoder()

    def preprocess(self, df):

        # Remove MongoDB ObjectId if present
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Handle missing values

        for column in df.columns:

            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())

            else:
                df[column] = df[column].fillna(df[column].mode()[0])

        # Encode purpose column
        df["purpose"] = self.encoder.fit_transform(df["purpose"])

        # Features
        X = df.drop(columns=["not.fully.paid"])

        # Target
        y = df["not.fully.paid"]

        # Scale numerical columns
        numerical_columns = X.columns

        X[numerical_columns] = self.scaler.fit_transform(
            X[numerical_columns]
        )

        # Create models folder if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save scaler and encoder
        joblib.dump(self.scaler, "models/scaler.pkl") #joblib saves the cleaned data 

        joblib.dump(self.encoder, "models/encoder.pkl")

        print("Preprocessing completed successfully.")

        return X, y
    
if __name__ == "__main__":

    from database.fetch_data import FetchData
    from structure.feature_engineering import FeatureEngineering

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    fetcher = FetchData()
    df = fetcher.get_dataframe()

    print("Original Shape:", df.shape)

    print("\nApplying Feature Engineering...")

    fe = FeatureEngineering()
    df = fe.create_features(df)

    print("Shape after Feature Engineering:", df.shape)

    print("\nApplying Preprocessing...")

    preprocessor = DataPreprocessor()

    X, y = preprocessor.preprocess(df)

    print("\nPreprocessing Successful!")
    print("X Shape :", X.shape)
    print("y Shape :", y.shape)

    print("\nFirst 5 rows of X:")
    print(X.head())

    print("\nFirst 5 values of y:")
    print(y.head())