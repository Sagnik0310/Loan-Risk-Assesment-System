from structure.prediction import LoanPrediction
from database.prediction_history import PredictionHistory

predictor = LoanPrediction()
history = PredictionHistory()

print("=" * 60)
print("LOAN RISK PREDICTION")
print("=" * 60)

# ---------------- Applicant Details ----------------

applicant = {

    "name": input("Applicant Name : ").strip(),

    "address": input("Address        : ").strip(),

    "phone": input("Mobile Number  : ").strip(),

    "email": input("Email          : ").strip()

}

# ---------------- Loan Details ----------------

loan_data = {

    "credit.policy": int(input("Credit Policy (1/0): ")),

    "purpose": input("Purpose: ").strip(),

    "int.rate": float(input("Interest Rate: ")),

    "installment": float(input("Installment: ")),

    "log.annual.inc": float(input("Log Annual Income: ")),

    "dti": float(input("Debt-To-Income Ratio: ")),

    "fico": int(input("FICO Score: ")),

    "days.with.cr.line": float(input("Days with Credit Line: ")),

    "revol.bal": float(input("Revolving Balance: ")),

    "revol.util": float(input("Revolving Utilization: ")),

    "inq.last.6mths": int(input("Inquiries Last 6 Months: ")),

    "delinq.2yrs": int(input("Delinquencies Last 2 Years: ")),

    "pub.rec": int(input("Public Records: "))

}

# ---------------- Prediction ----------------

result = predictor.predict(loan_data)

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

risk = "HIGH RISK" if result["Prediction"] == 1 else "LOW RISK"

print(f"Prediction : {risk}")
print(f"Probability: {result['Probability']:.2%}")

# ---------------- Save ----------------

history.save_prediction(

    applicant,

    loan_data,

    result

)

print("\nApplication saved successfully into MongoDB.")