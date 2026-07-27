from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.ml_engine.logger import logger

from src.ml_engine.config.settings import (
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    PROCESSED_DATASET_DIR,
)


TARGET = "fraud_label"



class FraudModelTrainer:


    def __init__(self):

        self.models = {

            "random_forest":

                RandomForestClassifier(

                    n_estimators=200,

                    random_state=RANDOM_STATE,

                    class_weight="balanced",

                    n_jobs=-1

                ),


            "xgboost":

                XGBClassifier(

                    n_estimators=300,

                    learning_rate=0.05,

                    max_depth=6,

                    subsample=0.8,

                    colsample_bytree=0.8,

                    eval_metric="logloss",

                    scale_pos_weight=10,

                    random_state=RANDOM_STATE

                )

        }



    def prepare_data(self, df):

        X = df.drop(
            columns=[TARGET]
        )

        y = df[TARGET]


        return train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )



    def train(self, X_train, y_train):

        results = {}

        best_model = None

        best_score = 0


        for name, model in self.models.items():

            logger.info(
                f"Training {name}"
            )


            model.fit(

                X_train,

                y_train

            )


            results[name] = model


        return results



    def evaluate(
        self,
        models,
        X_test,
        y_test
    ):


        best_name = None

        best_score = 0


        for name, model in models.items():


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


            logger.info(
                f"{name} | F1={f1:.4f} | AUC={auc:.4f}"
            )


            if f1 > best_score:

                best_score = f1

                best_name = name

                best_model = model



        logger.info(
            f"Best Model: {best_name}"
        )


        return best_model



    def save(self, model):

        MODEL_DIR.mkdir(
            exist_ok=True
        )


        path = (
            MODEL_DIR
            /
            "fraudec_best_model.pkl"
        )


        joblib.dump(
            model,
            path
        )


        logger.info(
            f"Model saved -> {path}"
        )




if __name__ == "__main__":


    dataset = (

        PROCESSED_DATASET_DIR

        /

        "fraudec_features.parquet"

    )


    df = pd.read_parquet(
        dataset
    )


    trainer = FraudModelTrainer()


    X_train, X_test, y_train, y_test = (

        trainer.prepare_data(df)

    )


    models = trainer.train(
        X_train,
        y_train
    )


    best_model = trainer.evaluate(
        models,
        X_test,
        y_test
    )


    trainer.save(
        best_model
    )