"""
Fraudec AI

PaySim Adapter

Purpose
-------
Loads the raw PaySim dataset

Performs

✓ validation
✓ cleaning
✓ datatype conversion
✓ missing value handling
✓ feature normalization

Outputs

datasets/normalized/paysim_normalized.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_engine.adapters.base_adapter import BaseAdapter
from src.ml_engine.logger import logger


class PaySimAdapter(BaseAdapter):

    REQUIRED_COLUMNS = [

        "step",

        "type",

        "amount",

        "nameOrig",

        "oldbalanceOrg",

        "newbalanceOrig",

        "nameDest",

        "oldbalanceDest",

        "newbalanceDest",

        "isFraud",

        "isFlaggedFraud"

    ]

    def __init__(self, dataset_path):

        super().__init__(dataset_path)

    # -----------------------------------------------------

    def validate(self, df):

        missing = [

            c

            for c in self.REQUIRED_COLUMNS

            if c not in df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing columns : {missing}"

            )

        logger.info("Validation successful.")

    # -----------------------------------------------------

    def clean(self, df):

        logger.info("Cleaning PaySim dataset...")

        df = self.standardize_columns(df)

        df = self.remove_duplicates(df)

        # remove impossible amounts

        df = df[df["amount"] >= 0]

        # replace negative balances

        balance_cols = [

            "oldbalanceorg",

            "newbalanceorig",

            "oldbalancedest",

            "newbalancedest"

        ]

        for col in balance_cols:

            df[col] = df[col].clip(lower=0)

        # missing transaction type

        df["type"] = (

            df["type"]

            .fillna("UNKNOWN")

            .str.upper()

        )

        # ids

        df["nameorig"] = (

            df["nameorig"]

            .fillna("UNKNOWN")

            .astype(str)

        )

        df["namedest"] = (

            df["namedest"]

            .fillna("UNKNOWN")

            .astype(str)

        )

        # fraud labels

        df["isfraud"] = (

            df["isfraud"]

            .astype(int)

        )

        df["isflaggedfraud"] = (

            df["isflaggedfraud"]

            .astype(int)

        )

        # numeric columns

        numeric = [

            "amount",

            "oldbalanceorg",

            "newbalanceorig",

            "oldbalancedest",

            "newbalancedest"

        ]

        for col in numeric:

            df[col] = (

                pd.to_numeric(

                    df[col],

                    errors="coerce"

                )

                .fillna(0)

            )

        return df

    # -----------------------------------------------------

    def engineer(self, df):

        logger.info("Engineering PaySim features...")

        df["amount_log"] = np.log1p(

            df["amount"]

        )

        df["origin_balance_change"] = (

            df["oldbalanceorg"]

            -

            df["newbalanceorig"]

        )

        df["destination_balance_change"] = (

            df["newbalancedest"]

            -

            df["oldbalancedest"]

        )

        df["balance_ratio"] = (

            df["amount"]

            /

            (

                df["oldbalanceorg"]

                +

                1

            )

        )

        df["high_value_transaction"] = (

            df["amount"]

            >

            df["amount"].quantile(.95)

        ).astype(int)

        # 1 step = 1 hour

        df["transaction_hour"] = (

            df["step"] % 24

        )

        df["transaction_day"] = (

            df["step"] // 24

        )

        return df

    # -----------------------------------------------------

    def normalize(self, df):

        logger.info("Normalizing columns...")

        rename = {

            "type": "transaction_type",

            "amount": "amount",

            "oldbalanceorg": "old_balance_origin",

            "newbalanceorig": "new_balance_origin",

            "oldbalancedest": "old_balance_destination",

            "newbalancedest": "new_balance_destination",

            "nameorig": "customer_id",

            "namedest": "merchant_id",

            "isfraud": "fraud_label",

            "isflaggedfraud": "flagged_fraud"

        }

        df = df.rename(

            columns=rename

        )

        return df

    # -----------------------------------------------------

    def process(

        self,

        output_path

    ):

        logger.info("=" * 60)

        logger.info("Processing PaySim")

        logger.info("=" * 60)

        df = self.load()

        self.validate(df)

        df = self.clean(df)

        df = self.engineer(df)

        df = self.normalize(df)

        self.save(

            df,

            output_path

        )

        logger.info(

            f"Finished : {output_path}"

        )
