from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()




model = joblib.load("loan_model.pkl")
threshold = joblib.load("threshold.pkl")

class LoanRequest(BaseModel):
    Gender: int
    Married: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float

    Property_Area: str
    Dependents: str

@app.post("/predict")
def predict(data: LoanRequest):

    df = pd.DataFrame([{
        "Gender": data.Gender,
        "Married": data.Married,
        "Education": data.Education,
        "Self_Employed": data.Self_Employed,
        "ApplicantIncome": data.ApplicantIncome,
        "CoapplicantIncome": data.CoapplicantIncome,
        "LoanAmount": data.LoanAmount,
        "Loan_Amount_Term": data.Loan_Amount_Term,
        "Credit_History": data.Credit_History,

        "Property_Area_Rural": 1 if data.Property_Area == "Rural" else 0,
        "Property_Area_Semiurban": 1 if data.Property_Area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if data.Property_Area == "Urban" else 0,

        "Dependents_0": 1 if data.Dependents == "0" else 0,
        "Dependents_1": 1 if data.Dependents == "1" else 0,
        "Dependents_2": 1 if data.Dependents == "2" else 0,
        "Dependents_3+": 1 if data.Dependents == "3+" else 0,
    }])

    probability = model.predict_proba(df)[0][1]

    prediction = int(probability >= threshold)

    if probability >= 0.8:
        risk = "Low Risk"
    elif probability >= 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return {
    "decision": "Approved" if prediction else "Rejected",
    "approval_probability": round(float(probability),4),
    "risk_level": risk
    }

