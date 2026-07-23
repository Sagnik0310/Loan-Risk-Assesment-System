import os   #file handling
import joblib

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

            if df[column].dtype == "object":

                df[column] = df[column].fillna(df[column].mode()[0])

            else:

                df[column] = df[column].fillna(df[column].median())

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