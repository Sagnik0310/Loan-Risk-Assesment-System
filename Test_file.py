from structure.prediction import LoanPrediction

predictor = LoanPrediction()

print("=" * 60)
print("LOAN RISK PREDICTION")
print("=" * 60)

data = {
    "credit.policy": int(input("Credit Policy (1/0): ")),
    "purpose": input("Purpose: "),
    "int.rate": float(input("Interest Rate: ")),
    "installment": float(input("Installment: ")),
    "log.annual.inc": float(input("Log Annual Income: ")),
    "dti": float(input("Debt to Income Ratio: ")),
    "fico": int(input("FICO Score: ")),
    "days.with.cr.line": float(input("Days with Credit Line: ")),
    "revol.bal": float(input("Revolving Balance: ")),
    "revol.util": float(input("Revolving Utilization (%): ")),
    "inq.last.6mths": int(input("Inquiries in Last 6 Months: ")),
    "delinq.2yrs": int(input("Delinquencies in Last 2 Years: ")),
    "pub.rec": int(input("Public Records: "))
}

result = predictor.predict(data)

print("\n========== RESULT ==========")

if result["Prediction"] == 1:
    print("Prediction : Loan will NOT be fully paid (High Risk)")
else:
    print("Prediction : Loan is likely to be fully paid (Low Risk)")

print(f"Probability of Default : {result['Probability']:.2%}")