from fastapi import FastAPI
from pydantic import BaseModel
from src.ml_engine.fraud_agent import FraudInvestigator

from src.ml_engine.ensemble import FraudEnsemble

app = FastAPI(
    title="Fraudec API",
    version="1.0"
)

ensemble = FraudEnsemble()

class TransactionRequest(BaseModel):

    # PaySim
    amount: float
    transaction_type: str
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    # IEEE
    card_network: str
    card_type: str
    payer_email: str
    receiver_email: str
    device_type: str
    device_info: str

    transaction_hour: int

    strictness: float = 50

@app.get("/")
def home():

    return {
        "message": "Fraudec API Running"
    }

@app.post("/predict")
def predict(request: TransactionRequest):

    transaction = request.model_dump()

    strictness = transaction.pop("strictness")

    result = ensemble.predict(
    transaction,
    strictness
    )
    investigator = FraudInvestigator()

    report = investigator.generate_report(
        transaction,
        result
    )

    result["analysis"] = report

    return result

