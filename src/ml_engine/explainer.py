
import joblib

import pandas as pd

import shap

from src.ml_engine.config.settings import MODEL_DIR

from src.ml_engine.logger import logger



class FraudExplainer:


    def __init__(self):

        model_path = (

            MODEL_DIR

            /

            "fraudec_best_model.pkl"

        )

        self.model = joblib.load(
            model_path
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )


    def explain(
        self,
        transaction: pd.DataFrame
    ):

        logger.info(
            "Generating SHAP explanation"
        )


        shap_values = self.explainer.shap_values(
            transaction
        )


        prediction = self.model.predict(
            transaction
        )[0]


        probability = self.model.predict_proba(
            transaction
        )[0][1]


        explanation = {


            "prediction":

                "Fraud"
                if prediction == 1
                else
                "Normal",


            "fraud_probability":

                float(probability),


            "feature_contribution":

                dict(
                    zip(
                        transaction.columns,
                        shap_values[0]
                    )
                )

        }


        return explanation