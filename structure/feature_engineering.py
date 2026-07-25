import pandas as pd


class FeatureEngineering:

    def __init__(self):
        pass

    def create_features(self, df):

        # Credit history in years
        df["credit_history_years"] = (
            df["days.with.cr.line"] / 365
        )

        # Revolving balance per FICO score
        df["revol_balance_per_fico"] = (
            df["revol.bal"] / (df["fico"] + 1)
        )

        # Total inquiry risk
        df["inquiry_risk"] = (
            df["inq.last.6mths"] +
            df["delinq.2yrs"]
        )

        # Convert utilization percentage to decimal
        df["credit_utilization_ratio"] = (
            df["revol.util"] / 100
        )

        # Interest burden
        df["interest_installment"] = (
            df["int.rate"] *
            df["installment"]
        )

        return df
if __name__ == "__main__":

    from database.fetch_data import FetchData

    fetcher = FetchData()

    df = fetcher.fetch_data()

    fe = FeatureEngineering()

    df = fe.create_features(df)

    print(df.head())

    print(df.columns)