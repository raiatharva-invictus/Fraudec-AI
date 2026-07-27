import streamlit as st
import requests

API = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fraudec AI",
    layout="wide"
)

st.title("🛡️ Fraudec AI")
st.caption("AI Powered Financial Fraud Detection System")

st.divider()

with st.form("prediction_form"):

    amount = st.number_input("Amount", min_value=0.0)

    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "TRANSFER",
            "CASH_OUT",
            "PAYMENT",
            "DEBIT",
            "CASH_IN"
        ]
    )

    oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)

    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)

    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)

    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

    card_network = st.selectbox(
        "Card Network",
        [
            "visa",
            "mastercard",
            "discover",
            "amex",
            "other"
        ]
    )

    card_type = st.selectbox(
        "Card Type",
        [
            "debit",
            "credit",
            "prepaid"
        ]
    )

    payer_email = st.text_input("Payer Email Domain")

    receiver_email = st.text_input("Receiver Email Domain")

    device_type = st.selectbox(
        "Device Type",
        [
            "mobile",
            "desktop",
            "tablet",
            "other"
        ]
    )

    device_info = st.text_input("Device Information")

    transaction_hour = st.slider(
        "Transaction Hour",
        0,
        23,
        12
    )

    strictness = st.slider(
        "Detection Strictness",
        min_value=0,
        max_value=100,
        value=50
    )

    submit = st.form_submit_button("Predict")

if submit:

    payload = {

        "amount": amount,
        "transaction_type": transaction_type,

        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,

        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,

        "card_network": card_network,
        "card_type": card_type,

        "payer_email": payer_email,
        "receiver_email": receiver_email,

        "device_type": device_type,
        "device_info": device_info,

        "transaction_hour": transaction_hour,

        "strictness": strictness
    }

    try:

        with st.spinner("Analyzing transaction..."):

            response = requests.post(
                API,
                json=payload
            )
            response.raise_for_status()

        result = response.json()
        st.write(result)

    except Exception as e:

        st.error(f"Prediction failed: {e}")

        st.write(
            response.text
        )

        st.stop()
    st.divider()

    if result["prediction"] == "Fraud":

        st.error("🚨 Fraud Detected")

    else:

        st.success("✅ Legitimate Transaction")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Risk Score",
        f"{result['risk_score']*100:.2f}%"
    )

    c2.metric(
        "Confidence",
        f"{result['confidence']:.2f}%"
    )

    c3.metric(
        "Threshold",
        result["threshold"]
    )
    st.progress(float(result["risk_score"]))

    st.caption(
        f"Overall Fraud Probability : {result['risk_score']*100:.2f}%"
    )

    st.divider()

    st.subheader("🤖 AI Fraud Analyst")

    st.write(result["analysis"])
    st.markdown(result["analysis"])

    with st.expander("Advanced Technical Details"):

        st.write(
            f"PaySim Probability : {result['paysim_probability']:.2%}"
        )

        st.write(
            f"IEEE Probability : {result['ieee_probability']:.2%}"
        )

        st.write(
            f"Decision Threshold : {result['threshold']}"
        )