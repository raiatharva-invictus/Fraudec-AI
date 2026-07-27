from pathlib import Path
import numpy as np
import pandas as pd

from src.ml_engine.logger import logger


class FraudFeatureEngineer:


    def create_features(self, df):

        logger.info(
            "Starting feature engineering..."
        )

        df = df.copy()

        pca_columns = [

        col for col in df.columns

        if col.startswith("cc_v")

        ]


        df.drop(

            columns=pca_columns,

            inplace=True,

            errors="ignore"

        )


        logger.info(

            f"Removed {len(pca_columns)} PCA credit card features"

        )


        # -----------------------------
        # Transaction Amount Features
        # -----------------------------

        if "amount" in df.columns:

            df["amount_log"] = np.log1p(
                df["amount"]
            )

            df["amount_high_risk"] = (
                df["amount"]
                >
                df["amount"].quantile(0.95)
            ).astype(int)


        # -----------------------------
        # Balance Features (PaySim)
        # -----------------------------

        if (
            "old_balance" in df.columns
            and
            "new_balance" in df.columns
        ):

            df["balance_change"] = (

                df["new_balance"]
                -
                df["old_balance"]

            )


            df["balance_ratio"] = (

                df["new_balance"]

                /

                (
                    df["old_balance"]
                    + 1
                )

            )


        # -----------------------------
        # Time Features
        # -----------------------------

        if "timestamp" in df.columns:

            try:

                timestamp = pd.to_datetime(
                    df["timestamp"]
                )

                df["transaction_hour"] = (
                    timestamp.dt.hour
                )

                df["transaction_day"] = (
                    timestamp.dt.day
                )

                df["transaction_month"] = (
                    timestamp.dt.month
                )

            except Exception:

                logger.info(
                    "Timestamp conversion skipped."
                )

        if "dataset_source" in df.columns:

            df["dataset_source_risk"] = (
                df["dataset_source"]
                .astype("category")
                .cat.codes
            )

        df["missing_fields_count"] = (

            df.isna()
            .sum(axis=1)

        )


        logger.info(
            "Feature engineering completed."
        )

        if "Class" in df.columns:

            df.rename(

                columns={
                    "Class":"fraud_label"
                },

                inplace=True

            )


        if "isFraud" in df.columns:

            df.rename(

                columns={
                    "isFraud":"fraud_label"
                },

                inplace=True

            )


        return df



    def transform(
        self,
        input_path,
        output_path
    ):

        df = pd.read_parquet(
            input_path
        )


        df = self.create_features(
            df
        )


        df.to_parquet(
            output_path,
            index=False
        )


        logger.info(
            f"Features saved -> {output_path}"
        )



if __name__ == "__main__":


    from src.ml_engine.config.settings import (
        PROCESSED_DATASET_DIR
    )


    engineer = FraudFeatureEngineer()


    engineer.transform(

        input_path=(

            PROCESSED_DATASET_DIR

            /

            "fraudec_processed.parquet"

        ),

        output_path=(

            PROCESSED_DATASET_DIR

            /

            "fraudec_features.parquet"

        )

    )