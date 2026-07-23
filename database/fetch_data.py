import pandas as pd

from database.mongodb import MongoDBConnection
from database.db_collections import Collections


class FetchData:

    def __init__(self):
        self.mongo = MongoDBConnection()

    def get_dataframe(self):
        

        try:
            # Connect to MongoDB
            db = self.mongo.connect()

            # Select collection
            collection = db[Collections.LOAN_DATA]

            # Fetch all documents
            data = list(collection.find())

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Remove MongoDB ObjectId column
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

        finally:
            self.mongo.close()


if __name__ == "__main__":

    fetcher = FetchData()

    df = fetcher.get_dataframe()

    print(df.head())
    print("\nShape :", df.shape)
    print("\nColumns :")
    print(df.columns.tolist())