from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.ml_engine.logger import logger


class BaseAdapter(ABC):

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

    def load(self):

        logger.info(f"Loading {self.dataset_path}")

        if self.dataset_path.suffix == ".csv":
            return pd.read_csv(self.dataset_path)

        if self.dataset_path.suffix == ".parquet":
            return pd.read_parquet(self.dataset_path)

        raise ValueError("Unsupported dataset")

    def remove_duplicates(self, df):

        before = len(df)

        df = df.drop_duplicates()

        logger.info(
            f"Removed {before-len(df)} duplicates"
        )

        return df

    def standardize_columns(self, df):

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        return df

    def convert_datetime(self, df, columns):

        for col in columns:

            if col in df.columns:

                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

        return df

    def save(self, df, output):

        output = Path(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_parquet(
            output,
            index=False
        )

        logger.info(
            f"Saved {output}"
        )

    @abstractmethod
    def process(self):
        pass