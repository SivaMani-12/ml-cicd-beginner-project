from pathlib import Path
import joblib,pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
app=FastAPI(title="Customer Churn Prediction API")
model=None
class Customer(BaseModel):
    tenure_months:int=Field(ge=0,le=100)
    monthly_charges:float=Field(ge=0,le=1000)
    support_tickets:int=Field(ge=0,le=100)
    contract_type:str
    uses_auto_pay:bool
@app.on_event("startup")
def load_model():
    global model
    if Path("artifacts/model.joblib").exists(): model=joblib.load("artifacts/model.joblib")
@app.get("/health")
def health(): return {"status":"ok","model_loaded":model is not None}
@app.post("/predict")
def predict(customer:Customer):
    if customer.contract_type not in {"month_to_month","one_year","two_year"}: raise HTTPException(422,"Invalid contract_type")
    if model is None: raise HTTPException(503,"Model missing; run python -m src.pipeline")
    p=float(model.predict_proba(pd.DataFrame([customer.model_dump()]))[0,1])
    return {"churn_probability":round(p,4),"predicted_churn":p>=.5}
