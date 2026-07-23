import os
from dotenv import load_dotenv

load_dotenv()


class Collections:
    LOAN_DATA = os.getenv("COLLECTION_NAME")