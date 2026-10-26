"""
Automated regression and UI verification test suite for LoanPredict.
Tests:
1. Exact ML baseline prediction preservation across representative profiles
2. Model artifact integrity and feature order / encoder class mapping
3. MongoDB CRUD round-trip (save, search, delete) with cleanup
4. DataFrame normalization, empty-state handling, and PII masking
5. Offline evaluation artifact loading
6. Streamlit UI page rendering and navigation without unintended DB writes
"""

import unittest
import joblib

from database.prediction_history import PredictionHistory
from structure.prediction import LoanPrediction
from ui.components import mask_email, mask_phone
from ui.services import (
    PURPOSE_OPTIONS,
    load_classification_reports,
    load_cross_validation_results,
    load_hyperparameter_results,
    load_model_comparison,
    prepare_history_dataframe,
)


class TestLoanPredictRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = LoanPrediction()
        cls.scaler = joblib.load("models/scaler.pkl")
        cls.encoder = joblib.load("models/encoder.pkl")

    def test_01_feature_order_and_encoder_classes(self):
        """Verify scaler feature names/order and encoder classes match UI mappings."""
        expected_features = [
            "credit.policy",
            "purpose",
            "int.rate",
            "installment",
            "log.annual.inc",
            "dti",
            "fico",
            "days.with.cr.line",
            "revol.bal",
            "revol.util",
            "inq.last.6mths",
            "delinq.2yrs",
            "pub.rec",
        ]
        self.assertEqual(list(self.scaler.feature_names_in_), expected_features)

        expected_classes = [
            "all_other",
            "credit_card",
            "debt_consolidation",
            "educational",
            "home_improvement",
            "major_purchase",
            "small_business",
        ]
        self.assertEqual(list(self.encoder.classes_), expected_classes)
        self.assertEqual(sorted(PURPOSE_OPTIONS.values()), expected_classes)

    def test_02_baseline_predictions_unchanged(self):
        """Verify representative inputs produce the exact baseline outputs."""
        sample_1 = {
            "credit.policy": 1,
            "purpose": "debt_consolidation",
            "int.rate": 0.11,
            "installment": 300.0,
            "log.annual.inc": 11.0,
            "dti": 15.0,
            "fico": 700,
            "days.with.cr.line": 3500.0,
            "revol.bal": 10000.0,
            "revol.util": 45.0,
            "inq.last.6mths": 1,
            "delinq.2yrs": 0,
            "pub.rec": 0,
        }
        res_1 = self.predictor.predict(sample_1)
        self.assertEqual(res_1["Prediction"], 0)
        self.assertAlmostEqual(res_1["Probability"], 0.11592416327365528, places=12)

        sample_2 = {
            "credit.policy": 0,
            "purpose": "small_business",
            "int.rate": 0.21,
            "installment": 850.0,
            "log.annual.inc": 9.5,
            "dti": 28.5,
            "fico": 615,
            "days.with.cr.line": 1100.0,
            "revol.bal": 45000.0,
            "revol.util": 95.0,
            "inq.last.6mths": 12,
            "delinq.2yrs": 3,
            "pub.rec": 2,
        }
        res_2 = self.predictor.predict(sample_2)
        self.assertEqual(res_2["Prediction"], 1)
        self.assertAlmostEqual(res_2["Probability"], 0.8074064836469612, places=12)

        sample_3 = {
            "credit.policy": 1,
            "purpose": "credit_card",
            "int.rate": 0.08,
            "installment": 180.0,
            "log.annual.inc": 11.6,
            "dti": 8.0,
            "fico": 780,
            "days.with.cr.line": 6200.0,
            "revol.bal": 3500.0,
            "revol.util": 15.0,
            "inq.last.6mths": 0,
            "delinq.2yrs": 0,
            "pub.rec": 0,
        }
        res_3 = self.predictor.predict(sample_3)
        self.assertEqual(res_3["Prediction"], 0)
        self.assertAlmostEqual(res_3["Probability"], 0.07278831735554267, places=12)

    def test_03_pii_masking_and_empty_dataframe(self):
        """Verify PII masking helpers and graceful handling of empty or partial records."""
        self.assertEqual(mask_email("jane.doe@example.com"), "ja***e@example.com")
        self.assertEqual(mask_email(""), "—")
        self.assertEqual(mask_phone("+1 (555) 234-5678"), "•••-•••-5678")
        self.assertEqual(mask_phone(""), "—")

        empty_df = prepare_history_dataframe([])
        self.assertTrue(empty_df.empty)

        partial_df = prepare_history_dataframe([{"prediction": 1, "probability": 0.75, "fico": 640}])
        self.assertEqual(len(partial_df), 1)
        self.assertEqual(partial_df.iloc[0]["risk_label"], "High Risk")
        self.assertEqual(partial_df.iloc[0]["prob_pct"], 75.0)

    def test_04_offline_evaluation_reports_load(self):
        """Verify offline evaluation CSV and JSON artifacts load accurately."""
        comp_df = load_model_comparison()
        cv_df = load_cross_validation_results()
        hp_df = load_hyperparameter_results()
        cls_reports = load_classification_reports()

        self.assertEqual(len(comp_df), 7)
        self.assertEqual(len(cv_df), 7)
        self.assertEqual(len(hp_df), 7)
        self.assertEqual(len(cls_reports), 7)

    def test_05_mongodb_crud_roundtrip(self):
        """Verify save, exact search, and delete on MongoDB leave no residual test records."""
        history = PredictionHistory()
        initial_count = history.total_applications()

        test_applicant = {
            "name": "__REGRESSION_TEST_USER__",
            "address": "100 Test Way",
            "phone": "999-000-9999",
            "email": "regression.test@loanpredict.internal",
        }
        test_loan = {
            "credit.policy": 1,
            "purpose": "debt_consolidation",
            "int.rate": 0.11,
            "installment": 300.0,
            "log.annual.inc": 11.0,
            "dti": 15.0,
            "fico": 700,
            "days.with.cr.line": 3500.0,
            "revol.bal": 10000.0,
            "revol.util": 45.0,
            "inq.last.6mths": 1,
            "delinq.2yrs": 0,
            "pub.rec": 0,
        }
        test_result = {"Prediction": 0, "Probability": 0.11592416327365528}

        inserted_id = history.save_prediction(test_applicant, test_loan, test_result)
        self.assertIsNotNone(inserted_id)

        try:
            found = history.search_application(
                test_applicant["name"], test_applicant["email"], test_applicant["phone"]
            )
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["prediction"], 0)
        finally:
            deleted = history.delete_application(
                test_applicant["name"], test_applicant["email"], test_applicant["phone"]
            )
            self.assertEqual(deleted, 1)

        final_count = history.total_applications()
        self.assertEqual(final_count, initial_count)

    def test_06_streamlit_ui_navigation_no_unintended_writes(self):
        """Verify Streamlit app renders all 5 pages cleanly without unintended DB writes."""
        from streamlit.testing.v1 import AppTest

        history = PredictionHistory()
        count_before = history.total_applications()

        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"AppTest initial run exception: {at.exception}")

        # Dismiss welcome modal or navigate across all 5 pages via session state
        for page_name in ["overview", "new_prediction", "history", "analytics", "model_insights"]:
            at.session_state["welcome_shown"] = True
            at.session_state["page"] = page_name
            at.run()
            self.assertEqual(
                len(at.exception), 0, f"Exception on page '{page_name}': {at.exception}"
            )

        # Toggle light mode and verify clean render
        at.session_state["dark_mode"] = False
        at.session_state["page"] = "overview"
        at.run()
        self.assertEqual(len(at.exception), 0, f"Exception in light mode: {at.exception}")

        count_after = history.total_applications()
        self.assertEqual(count_before, count_after, "Browsing pages must not write to MongoDB!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

