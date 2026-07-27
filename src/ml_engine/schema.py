"""
Fraudec Unified Transaction Schema
"""

from typing import Optional

from pydantic import BaseModel

from src.ml_engine.constants import DatasetSource


class TransactionRecord(BaseModel):

    transaction_id: str

    customer_id: Optional[str] = None

    merchant_id: Optional[str] = None

    amount: float

    payment_method: Optional[str] = None

    transaction_type: Optional[str] = None

    timestamp: Optional[str] = None

    fraud_label: int

    dataset_source: DatasetSource