import pandas as pd
from pymongo import MongoClient

client=MongoClient("mongodb+srv://Anantika:Anantika123@cluster0.gn7pxud.mongodb.net/?appName=Cluster0")

db=client["machine_learning"]

collection=db["loan_assessment"]

df=pd.read_csv("loan_data.csv")

collection.insert_many(df.to_dict(orient="records"))

print("successful")