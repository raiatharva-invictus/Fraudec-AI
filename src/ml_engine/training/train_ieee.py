import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score
)
from xgboost import XGBClassifier
from src.ml_engine.logger import logger


DATA_PATH = (
    "datasets/normalized/"
    "ieee_normalized.parquet"
)
MODEL_PATH = (
    "models/ieee_xgb.pkl"
)

def main():

    logger.info("Loading IEEE dataset")

    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=["fraud_label"])

    y = df["fraud_label"]

    drop_cols = [

        "transaction_id",
        "customer_id",
        "merchant_id"

    ]


    X = X.drop(
        columns=[
            c for c in drop_cols
            if c in X.columns
        ]
    )

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    FEATURE_PATH = "models/ieee_features.pkl" 

    joblib.dump(
        list(X.columns),
        FEATURE_PATH
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        stratify=y,

        random_state=42

    )

    fraud_weight = (

        y_train.value_counts()[0]
        /
        y_train.value_counts()[1]

    )

    logger.info(
        f"scale_pos_weight={fraud_weight}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    logger.info("Training IEEE XGBoost")

    model.fit(X_train, y_train)


    pred = model.predict(X_test)


    prob = model.predict_proba(X_test)[:,1]

    logger.info(
        classification_report(
            y_test,
            pred
        )
    )

    logger.info(
        f"F1: {f1_score(y_test,pred)}"
    )

    logger.info(
        f"AUC: {roc_auc_score(y_test,prob)}"
    )



    Path(
        "models"
    ).mkdir(
        exist_ok=True
    )


    joblib.dump(
        model,
        MODEL_PATH
    )


    logger.info(
        f"Saved {MODEL_PATH}"
    )



if __name__ == "__main__":
    main()