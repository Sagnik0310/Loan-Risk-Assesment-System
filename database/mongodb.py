from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

class MongoDBConnection:

    def __init__(self):
        self.uri = os.getenv("MONGO_URI")
        self.database_name = os.getenv("DATABASE_NAME")

        self.client = None
        self.db = None

    def connect(self):

        self.client = MongoClient(self.uri)

        self.db = self.client[self.database_name]

        return self.db

    def close(self):

        if self.client:
            self.client.close()