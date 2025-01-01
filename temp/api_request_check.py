import requests

#
# url = "http://localhost:8000/"
#
# try:
#     response: Response = requests.get(url)
#     if response.status_code == 200:
#         print("Request was successful!")
#         print("Response content:")
#         print(response.json())
#     else:
#         print(f"Request failed with status code: {response.status_code}")
# except requests.exceptions.RequestException as e:
#     print("An error occurred:", e)


# PREDICT_URL = "http://localhost:8000/predict"
PREDICT_URL = "https://churnobyl-api-service-968699229335.us-east1.run.app/predict"

print("\n\n")
print("#####" * 3)
print("\n\n\n")

try:
    # 7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
    payload = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
    }
    # payload = json.dumps(payload)
    headers = {"Content-Type": "application/json"}
    response = requests.post(PREDICT_URL, json=payload, headers=headers)
    if response.status_code == 200:
        print("Request was successful!")
        print("Response content:")
        print(response.json())
    else:
        print(f"Request failed with status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
