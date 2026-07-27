"""
Fraudec AI

Model Evaluation Pipeline
"""

import json

import joblib

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

from src.ml_engine.config.settings import (
    MODEL_DIR,
    PROCESSED_DATASET_DIR,
)

from src.ml_engine.logger import logger


TARGET = "fraud_label"



REPORT_DIR = "reports"



def main():

    import os

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )


    df = pd.read_parquet(

        PROCESSED_DATASET_DIR
        /
        "fraudec_features.parquet"

    )


    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )


    model = joblib.load(

        MODEL_DIR
        /
        "fraudec_best_model.pkl"

    )


    predictions = model.predict(
        X_test
    )


    probabilities = model.predict_proba(
        X_test
    )[:,1]


    f1 = f1_score(
        y_test,
        predictions
    )


    auc = roc_auc_score(
        y_test,
        probabilities
    )


    report = classification_report(

        y_test,

        predictions,

        output_dict=True

    )


    metrics = {

        "F1 Score": float(f1),

        "ROC AUC": float(auc),

    }


    with open(

        f"{REPORT_DIR}/metrics.json",

        "w"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4

        )


    with open(

        f"{REPORT_DIR}/classification_report.txt",

        "w"

    ) as file:

        file.write(

            classification_report(

                y_test,

                predictions

            )

        )


    matrix = confusion_matrix(

        y_test,

        predictions

    )


    plt.figure(

        figsize=(6,5)

    )

    plt.imshow(
        matrix
    )

    plt.title(
        "Fraudec Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.colorbar()


    plt.savefig(

        f"{REPORT_DIR}/confusion_matrix.png"

    )


    logger.info(
        "Evaluation completed."
    )



if __name__ == "__main__":

    main()