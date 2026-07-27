from __future__ import annotations
from typing import Iterable
import pandas as pd
from src.ml_engine.logger import logger

class ValidationError(Exception):
    """Raised when dataset validation fails."""

class DataValidator:
    """Utility class for validating datasets."""
    @staticmethod
    def require_columns(
        df: pd.DataFrame,
        required_columns: Iterable[str],
    ) -> None:
        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]
        if missing:
            raise ValidationError(
                f"Missing required columns: {missing}"
            )

    @staticmethod
    def ensure_not_empty(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValidationError(
                "Dataset is empty."
            )

    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        before = len(df)

        df = df.drop_duplicates()

        removed = before - len(df)

        if removed:
            logger.info(
                "Removed %d duplicate rows.",
                removed,
            )

        return df

    @staticmethod
    def missing_value_report(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        report = pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isnull().sum().values,
                "missing_percent": (
                    df.isnull().mean() * 100
                ).round(2).values,
            }
        )

        return report.sort_values(
            "missing_percent",
            ascending=False,
        )

    @staticmethod
    def class_distribution(
        df: pd.DataFrame,
        target_column: str,
    ) -> pd.Series:

        if target_column not in df.columns:
            raise ValidationError(
                f"{target_column} not found."
            )

        return (
            df[target_column]
            .value_counts(normalize=True)
            .sort_index()
        )

    @staticmethod
    def validate_numeric(
        df: pd.DataFrame,
        columns: Iterable[str],
    ) -> None:

        for column in columns:

            if column not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                raise ValidationError(
                    f"{column} must be numeric."
                )

    @staticmethod
    def basic_validation(
        df: pd.DataFrame,
        required_columns: Iterable[str],
    ) -> None:

        DataValidator.ensure_not_empty(df)

        DataValidator.require_columns(
            df,
            required_columns,
        )
        logger.info(
            "Dataset validation successful."
        )