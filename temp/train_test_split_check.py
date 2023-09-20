import pandas as pd
from sklearn import model_selection, preprocessing
import pickle as pkl

df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

X, y = df.drop(columns=["Churn"]), df[["Churn"]]
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y,
)
print(X_train["MultipleLines"].head(1))

CAT_COLS_OE = [
    "OnlineSecurity",
    "MultipleLines",
    "Dependents",
    "DeviceProtection",
    "StreamingTV",
    "OnlineBackup",
    "gender",
    "SeniorCitizen",
    "PhoneService",
    "Partner",
    "PaperlessBilling",
    "StreamingMovies",
    "TechSupport",
]

# encoder_oe = preprocessing.OrdinalEncoder()
# encoder_oe.fit(X_train[CAT_COLS_OE].values)

encoder_oe: preprocessing.OrdinalEncoder = pkl.load(
    open("./temp/artifacts/encoder_oe_.pkl", "rb")
)
print(encoder_oe.categories_)

X_train__transformed = encoder_oe.transform(X_train[CAT_COLS_OE].values)
X_train__transformed_df: pd.DataFrame = pd.DataFrame(
    X_train__transformed, columns=CAT_COLS_OE
)

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

payload: pd.DataFrame = pd.DataFrame(payload, index=[0])
out_payload = encoder_oe.transform(payload[CAT_COLS_OE].values)
print(out_payload)

print(encoder_oe.categories_)
