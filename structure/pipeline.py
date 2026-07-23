from database.fetch_data import FetchData
from structure.preprocessing import DataPreprocessor

fetcher = FetchData()

df = fetcher.get_dataframe()

preprocessor = DataPreprocessor()

X, y = preprocessor.preprocess(df)

print(X.head())

print(y.head())