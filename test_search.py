from database.prediction_history import PredictionHistory

history = PredictionHistory()

print("=" * 60)
print("SEARCH APPLICATION")
print("=" * 60)

name = input("Applicant Name : ").strip()

email = input("Email          : ").strip()

phone = input("Mobile Number  : ").strip()

records = history.search_application(

    name,

    email,

    phone

)

if not records:

    print("\nNo application found.")

else:

    print("\nApplication Found\n")

    for record in records:

        print("=" * 60)

        print(f"Name                  : {record['name']}")
        print(f"Address               : {record['address']}")
        print(f"Phone                 : {record['phone']}")
        print(f"Email                 : {record['email']}")

        print("-" * 60)

        print(f"Credit Policy         : {record['credit.policy']}")
        print(f"Purpose               : {record['purpose']}")
        print(f"Interest Rate         : {record['int.rate']}")
        print(f"Installment           : {record['installment']}")
        print(f"Log Annual Income     : {record['log.annual.inc']}")
        print(f"DTI                   : {record['dti']}")
        print(f"FICO                  : {record['fico']}")
        print(f"Days with Credit Line : {record['days.with.cr.line']}")
        print(f"Revolving Balance     : {record['revol.bal']}")
        print(f"Revolving Utilization : {record['revol.util']}")
        print(f"Inquiries             : {record['inq.last.6mths']}")
        print(f"Delinquencies         : {record['delinq.2yrs']}")
        print(f"Public Records        : {record['pub.rec']}")

        print("-" * 60)

        risk = "HIGH RISK" if record["prediction"] == 1 else "LOW RISK"

        print(f"Prediction            : {risk}")
        print(f"Probability           : {record['probability']:.2%}")
        print(f"Timestamp             : {record['timestamp']}")

        print("=" * 60)