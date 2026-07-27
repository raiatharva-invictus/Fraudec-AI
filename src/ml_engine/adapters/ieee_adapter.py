from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_engine.logger import logger


class IEEEAdapter:

    KEEP_TRANSACTION_COLUMNS = [

        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",

        "ProductCD",

        "card1",
        "card4",
        "card6",

        "addr1",

        "P_emaildomain",
        "R_emaildomain",

    ]


    KEEP_IDENTITY_COLUMNS = [

        "TransactionID",
        "DeviceType",
        "DeviceInfo"

    ]


    def __init__(
        self,
        transaction_path,
        identity_path
    ):

        self.transaction_path = Path(
            transaction_path
        )

        self.identity_path = Path(
            identity_path
        )


    # ---------------------------------------------

    def load(self):

        logger.info(
            "Loading IEEE transaction dataset"
        )

        transaction = pd.read_csv(
            self.transaction_path
        )


        logger.info(
            "Loading IEEE identity dataset"
        )

        identity = pd.read_csv(
            self.identity_path
        )


        return transaction, identity


    # ---------------------------------------------

    def merge(self, transaction, identity):

        logger.info("Merging transaction and identity")

        df = transaction.merge(
            identity,
            on="TransactionID",
            how="left",
            suffixes=("", "_identity")
        )

        if "TransactionID_identity" in df.columns:
            df = df.drop(
                columns=["TransactionID_identity"]
            )

        logger.info(
            f"Merged shape : {df.shape}"
        )

        return df



    # ---------------------------------------------

    def clean(self, df):

        logger.info(
            "Cleaning IEEE dataset"
        )


        df = df.drop_duplicates()
        df = df.loc[:, ~df.columns.duplicated()]


        available = [

            c

            for c in (

                self.KEEP_TRANSACTION_COLUMNS
                +
                self.KEEP_IDENTITY_COLUMNS

            )

            if c in df.columns

        ]


        df = df[available]


        # missing values

        categorical = [

            "ProductCD",

            "card4",

            "card6",

            "P_emaildomain",

            "R_emaildomain",

            "DeviceType",

            "DeviceInfo"

        ]


        for col in categorical:

            if col in df.columns:

                df[col] = (

                    df[col]

                    .fillna("UNKNOWN")

                    .astype(str)

                )


        numeric = [

            "TransactionAmt",

            "card1",

            "addr1"

        ]


        for col in numeric:

            if col in df.columns:

                df[col] = (

                    pd.to_numeric(

                        df[col],

                        errors="coerce"

                    )

                    .fillna(0)

                )


        df["isFraud"] = (

            df["isFraud"]

            .astype(int)

        )


        return df



    # ---------------------------------------------

    def feature_engineering(self, df):

        logger.info(
            "Creating IEEE features"
        )


        df["amount_log"] = np.log1p(

            df["TransactionAmt"]

        )


        df["transaction_hour"] = (

            (df["TransactionDT"] // 3600)

            % 24

        )


        df["transaction_day"] = (

            df["TransactionDT"]

            //

            86400

        )


        df["device_missing"] = (

            df["DeviceInfo"]

            ==

            "UNKNOWN"

        ).astype(int)



        df["email_missing"] = (

            df["P_emaildomain"]

            ==

            "UNKNOWN"

        ).astype(int)



        df["card_missing"] = (

            df["card4"]

            ==

            "UNKNOWN"

        ).astype(int)


        return df



    # ---------------------------------------------

    def normalize(self, df):

        logger.info(
            "Normalizing IEEE columns"
        )


        rename = {


            "TransactionID":
            "transaction_id",


            "isFraud":
            "fraud_label",


            "TransactionAmt":
            "amount",


            "ProductCD":
            "product_code",


            "DeviceType":
            "device_type",


            "DeviceInfo":
            "device_info",


            "P_emaildomain":
            "payer_email",


            "R_emaildomain":
            "receiver_email",


            "card4":
            "card_network",


            "card6":
            "card_type"

        }


        df = df.rename(
            columns=rename
        )


        return df



    # ---------------------------------------------

    def process(self, output):


        transaction, identity = self.load()


        df = self.merge(
            transaction,
            identity
        )


        df = self.clean(df)


        df = self.feature_engineering(df)


        df = self.normalize(df)

        df = df.loc[:, ~df.columns.duplicated()]

        Path(output).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        
        df.to_parquet(
            output,
            index=False
        )


        logger.info(
            f"Saved IEEE dataset -> {output}"
        )