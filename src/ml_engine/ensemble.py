import joblib
import pandas as pd


class FraudEnsemble:

    def __init__(self):

        self.paysim_model = joblib.load(
            "models/paysim_xgb.pkl"
        )

        self.ieee_model = joblib.load(
            "models/ieee_xgb.pkl"
        )

        self.paysim_features = joblib.load(
            "models/paysim_features.pkl"
        )

        self.ieee_features = joblib.load(
            "models/ieee_features.pkl"
        )


    def prepare(self, transaction, feature_list):

        df = pd.DataFrame([transaction])

        df = pd.get_dummies(df)

        df = df.reindex(
            columns=feature_list,
            fill_value=0
        )

        return df


    def predict(
        self,
        transaction,
        strictness="medium"
    ):

        pay_df = self.prepare(
            transaction,
            self.paysim_features
        )

        ieee_df = self.prepare(
            transaction,
            self.ieee_features
        )

        pay_prob = float(
            self.paysim_model.predict_proba(
                pay_df
            )[0][1]
        )

        ieee_prob = float(
            self.ieee_model.predict_proba(
                ieee_df
            )[0][1]
        )

        risk = (
            pay_prob * 0.5 +
            ieee_prob * 0.5
        )

        threshold = 0.7 - (strictness / 100) * 0.4

        prediction = (
            "Fraud"
            if risk >= threshold
            else "Legitimate"
        )

        return {
            "prediction": prediction,
            "risk_score": round(risk, 4),
            "confidence": round(max(risk, 1 - risk) * 100, 2),
            "paysim_probability": round(pay_prob, 4),
            "ieee_probability": round(ieee_prob, 4),
            "threshold": threshold
        }