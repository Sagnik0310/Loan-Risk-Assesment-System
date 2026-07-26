from datetime import datetime

from database.mongodb import MongoDBConnection
from database.db_collections import Collections


class PredictionHistory:

    def __init__(self):

        self.db = MongoDBConnection().connect()

        self.collection = self.db[Collections.PREDICTION_HISTORY]

    # -------------------------------------------------------
    # Save Prediction
    # -------------------------------------------------------

    def save_prediction(self, applicant, loan_data, result):

        document = {

            "timestamp": datetime.now(),

            "name": applicant["name"],
            "address": applicant["address"],
            "phone": applicant["phone"],
            "email": applicant["email"],

            **loan_data,

            "prediction": result["Prediction"],
            "probability": result["Probability"]

        }

        inserted = self.collection.insert_one(document)

        return inserted.inserted_id

    # -------------------------------------------------------
    # View All Applications
    # -------------------------------------------------------

    def get_history(self):

        return list(

            self.collection.find(
                {},
                {"_id": 0}
            ).sort("timestamp", -1)

        )

    # -------------------------------------------------------
    # Search Application
    # Name + Email + Mobile Number
    # -------------------------------------------------------

    def search_application(self, name, email, phone):

        return list(

            self.collection.find(

                {

                    "name": {
                        "$regex": f"^{name}$",
                        "$options": "i"
                    },

                    "email": {
                        "$regex": f"^{email}$",
                        "$options": "i"
                    },

                    "phone": phone

                },

                {"_id": 0}

            )

        )

    # -------------------------------------------------------
    # Delete One Application
    # -------------------------------------------------------

    def delete_application(self, name, email, phone):

        result = self.collection.delete_one(

            {

                "name": {
                    "$regex": f"^{name}$",
                    "$options": "i"
                },

                "email": {
                    "$regex": f"^{email}$",
                    "$options": "i"
                },

                "phone": phone

            }

        )

        return result.deleted_count

    # -------------------------------------------------------
    # Clear Entire History
    # -------------------------------------------------------

    def clear_history(self):

        result = self.collection.delete_many({})

        return result.deleted_count

    # -------------------------------------------------------
    # Total Applications
    # -------------------------------------------------------

    def total_applications(self):

        return self.collection.count_documents({})