import numpy as np
import pandas as pd
FEATURES=["tenure_months","monthly_charges","support_tickets","contract_type","uses_auto_pay"]
TARGET="churned"
def make_dataset(n=1200, seed=42):
    rng=np.random.default_rng(seed); tenure=rng.integers(1,73,n); charges=np.round(rng.uniform(25,125,n),2); tickets=rng.poisson(1.8,n).clip(0,10)
    contract=rng.choice(["month_to_month","one_year","two_year"],n,p=[.55,.27,.18]); autopay=rng.choice([True,False],n,p=[.62,.38])
    score=1.3*(contract=="month_to_month")+.18*tickets+.012*(charges-70)-.035*tenure-.55*autopay+rng.normal(0,.65,n)
    p=1/(1+np.exp(-score)); y=(rng.random(n)<p).astype(int)
    return pd.DataFrame({"tenure_months":tenure,"monthly_charges":charges,"support_tickets":tickets,"contract_type":contract,"uses_auto_pay":autopay,"churned":y})
def validate(df):
    expected=set(FEATURES+[TARGET])
    if set(df.columns)!=expected: raise ValueError(f"Column mismatch: expected {expected}")
    if df.empty: raise ValueError("Dataset is empty")
    if df.isnull().any().any(): raise ValueError("Dataset contains missing values")
    if not set(df[TARGET].unique()).issubset({0,1}): raise ValueError("Target must contain only 0 and 1")
    if not set(df.contract_type.unique()).issubset({"month_to_month","one_year","two_year"}): raise ValueError("Unexpected contract type")
    if (df[["tenure_months","monthly_charges","support_tickets"]]<0).any().any(): raise ValueError("Numeric features cannot be negative")
