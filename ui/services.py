"""
Service layer bridging the Streamlit UI with the existing ML prediction pipeline,
MongoDB prediction history, and offline evaluation report artifacts.

IMPORTANT: This module strictly calls the existing LoanPrediction and PredictionHistory
classes without modifying any ML logic, preprocessing, feature order, or thresholds.
"""

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

from database.prediction_history import PredictionHistory
from structure.prediction import LoanPrediction

warnings.filterwarnings("ignore", category=UserWarning)


# Exact 7 LendingClub purpose categories supported by models/encoder.pkl
PURPOSE_OPTIONS: Dict[str, str] = {
    "Debt Consolidation": "debt_consolidation",
    "Credit Card": "credit_card",
    "Home Improvement": "home_improvement",
    "Major Purchase": "major_purchase",
    "Small Business": "small_business",
    "Educational": "educational",
    "All Other": "all_other",
}

PURPOSE_LABELS: Dict[str, str] = {v: k for k, v in PURPOSE_OPTIONS.items()}


@st.cache_resource(show_spinner=False)
def get_predictor() -> LoanPrediction:
    """Load and cache the LoanPrediction instance once across sessions."""
    return LoanPrediction()


@st.cache_resource(show_spinner=False)
def get_encoder_classes() -> List[str]:
    """Return the exact classes fitted in models/encoder.pkl."""
    try:
        encoder = joblib.load("models/encoder.pkl")
        return list(encoder.classes_)
    except Exception:
        return list(PURPOSE_OPTIONS.values())


def get_history_service() -> Tuple[Optional[PredictionHistory], Optional[str]]:
    """
    Obtain a PredictionHistory instance from session state or initialize one safely.
    Returns (service, error_message).
    """
    if "history_service" not in st.session_state or st.session_state.history_service is None:
        try:
            st.session_state.history_service = PredictionHistory()
            st.session_state.db_error = None
        except Exception as exc:
            st.session_state.history_service = None
            st.session_state.db_error = f"Database connection unavailable: {exc}"
    return st.session_state.history_service, st.session_state.get("db_error")


def invalidate_history_cache() -> None:
    """Invalidate cached history records in session state after write/delete operations."""
    st.session_state.pop("cached_history_records", None)
    st.session_state.pop("cached_history_ts", None)


def fetch_history_records(force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch historical prediction records from MongoDB using PredictionHistory.get_history().
    Caches results in session state to avoid redundant database queries on page switches.
    """
    if not force_refresh and "cached_history_records" in st.session_state:
        return st.session_state.cached_history_records, st.session_state.get("db_error")

    service, err = get_history_service()
    if service is None:
        return [], err

    try:
        records = service.get_history()
        st.session_state.cached_history_records = records
        st.session_state.db_error = None
        return records, None
    except Exception as exc:
        err_msg = "Unable to retrieve application history from MongoDB."
        st.session_state.db_error = err_msg
        return [], err_msg


def save_application_prediction(
    applicant: Dict[str, str],
    loan_data: Dict[str, Any],
    result: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Persist a prediction using PredictionHistory.save_prediction() and refresh cache.
    Returns (success, error_message).
    """
    service, err = get_history_service()
    if service is None:
        return False, err or "Database connection unavailable."

    try:
        inserted_id = service.save_prediction(applicant, loan_data, result)
        invalidate_history_cache()
        return inserted_id is not None, None
    except Exception:
        return False, "Prediction was generated, but saving to MongoDB failed."


def search_applications_exact(
    name: str, email: str, phone: str
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Run the existing PredictionHistory.search_application() query."""
    service, err = get_history_service()
    if service is None:
        return [], err or "Database connection unavailable."

    try:
        results = service.search_application(name.strip(), email.strip(), phone.strip())
        return results, None
    except Exception:
        return [], "Search query failed. Please verify database connectivity."


def delete_application_record(
    name: str, email: str, phone: str
) -> Tuple[int, Optional[str]]:
    """Delete a single application record using PredictionHistory.delete_application()."""
    service, err = get_history_service()
    if service is None:
        return 0, err or "Database connection unavailable."

    try:
        deleted_count = service.delete_application(name, email, phone)
        if deleted_count > 0:
            invalidate_history_cache()
        return deleted_count, None
    except Exception:
        return 0, "Failed to delete the application record."


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Load offline model comparison metrics from reports/model_comparison.csv."""
    path = os.path.join("reports", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cross_validation_results() -> pd.DataFrame:
    """Load 5-fold cross-validation metrics from reports/cross_validation_results.csv."""
    path = os.path.join("reports", "cross_validation_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_hyperparameter_results() -> pd.DataFrame:
    """Load RandomizedSearchCV tuning summary from reports/hyperparameter_results.csv."""
    path = os.path.join("reports", "hyperparameter_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_classification_reports() -> Dict[str, Dict[str, Any]]:
    """Load per-model JSON classification reports from reports/."""
    report_files = {
        "Logistic Regression": "logistic_regression_classification_report.json",
        "Random Forest": "random_forest_classification_report.json",
        "Naive Bayes": "naive_bayes_classification_report.json",
        "XGBoost": "xgboost_classification_report.json",
        "K Nearest Neighbors": "k_nearest_neighbors_classification_report.json",
        "Support Vector Machine": "support_vector_machine_classification_report.json",
        "Decision Tree": "decision_tree_classification_report.json",
    }
    reports: Dict[str, Dict[str, Any]] = {}
    for model_name, filename in report_files.items():
        path = os.path.join("reports", filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reports[model_name] = json.load(f)
            except Exception:
                pass
    return reports


def prepare_history_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw MongoDB history records into a clean Pandas DataFrame
    with consistent types and human-readable labels.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).copy()

    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp_str"] = df["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M")
        df["date_only"] = df["timestamp_dt"].dt.date
    else:
        df["timestamp_dt"] = pd.NaT
        df["timestamp_str"] = "—"
        df["date_only"] = None

    for str_col in ["name", "email", "phone", "address", "purpose"]:
        if str_col not in df.columns:
            df[str_col] = ""
        df[str_col] = df[str_col].fillna("").astype(str)

    df["display_name"] = df["name"].apply(lambda x: x.strip() if x.strip() else "Unnamed Applicant")
    df["purpose_label"] = df["purpose"].apply(
        lambda p: PURPOSE_LABELS.get(p, p.replace("_", " ").title() if p else "Unspecified")
    )

    numeric_defaults = {
        "credit.policy": 1,
        "int.rate": 0.0,
        "installment": 0.0,
        "log.annual.inc": 0.0,
        "dti": 0.0,
        "fico": 0,
        "days.with.cr.line": 0.0,
        "revol.bal": 0.0,
        "revol.util": 0.0,
        "inq.last.6mths": 0,
        "delinq.2yrs": 0,
        "pub.rec": 0,
        "prediction": 0,
        "probability": 0.0,
    }
    for col, default_val in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val)

    # Maintain exact semantics: prediction == 0 -> Low Risk (Class 0), prediction == 1 -> High Risk (Class 1)
    df["risk_label"] = df["prediction"].map({0: "Low Risk", 1: "High Risk"}).fillna("Unknown")
    df["legacy_result"] = df["prediction"].map({0: "Approved", 1: "Denied"}).fillna("Unknown")
    df["prob_pct"] = (df["probability"] * 100.0).round(2)

    return df

