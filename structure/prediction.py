import joblib
import pandas as pd


class LoanPrediction:

    def __init__(self):

        self.model = joblib.load(
            "models/stacking_classifier.pkl"
        )

        self.scaler = joblib.load(
            "models/scaler.pkl"
        )

        self.encoder = joblib.load(
            "models/encoder.pkl"
        )

    def preprocess_input(self, data):

        df = pd.DataFrame([data])

        if "purpose" in df.columns:

            df["purpose"] = self.encoder.transform(
                df["purpose"]
            )

        df = self.scaler.transform(df)

        return df

    def predict(self, data):

        processed_data = self.preprocess_input(data)

        prediction = self.model.predict(
            processed_data
        )[0]

        probability = self.model.predict_proba(
            processed_data
        )[0][1]

        return {

            "Prediction": int(prediction),

            "Probability": float(probability)

        }