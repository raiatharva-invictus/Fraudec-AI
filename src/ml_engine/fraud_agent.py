import os
from src.ml_engine.logger import logger

class FraudInvestigator:


    def __init__(self):

        self.llm = None

        self.initialize_llm()

    def initialize_llm(self):
        try:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY")
            )
            logger.info(
                "LLM initialized"
            )
        except Exception as e:

            logger.warning(

                f"LLM unavailable: {e}"

            )

            self.llm = None

    def generate_report(
        self,
        transaction: dict,
        prediction_data: dict
    ):
        prompt = f"""
        You are Fraudec AI, an expert financial fraud analyst.

        Analyze the following transaction.

        Transaction Details:
        {transaction}

        Prediction Results:
        {prediction_data}

        Your task is to explain the model's decision to a banking analyst.

        Return your answer in the following format exactly:

        Summary:
        <short summary>

        Risk Level:
        <Low / Medium / High>

        Reasoning:
        - point 1
        - point 2
        - point 3

        Recommendation:
        <recommended action>

        Keep the explanation professional, concise, and avoid making claims not supported by the transaction data.
        """

        if self.llm:


            response = self.llm.invoke(
                prompt
            )
            return response.content
        else:

            probability = prediction_data.get("risk_score",0)

            if probability > 0.75:
                risk = "HIGH"
            elif probability > 0.4:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            return f"""
            Fraudec Investigation Report

            Risk Level:
            {risk}

            Prediction:
            {prediction_data.get("prediction")}

            Overall Risk Score:
            {probability:.2%}

            PaySim Model:
            {prediction_data.get("paysim_probability",0):.2%}

            IEEE Model:
            {prediction_data.get("ieee_probability",0):.2%}

            Recommendation:

            - Review the transaction before approval.
            - Verify customer identity.
            - Request additional authentication if required.
            """
