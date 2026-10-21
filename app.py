import streamlit as st
import pandas as pd
import joblib

# Load trained model and fitted preprocessor
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Prediction")

st.write(
    "Predict whether a customer is likely to subscribe "
    "to a term deposit using a Random Forest classifier."
)

st.divider()

with st.form("customer_form"):

    # PERSONAL DETAILS
    st.subheader("👤 Personal Details")

    c = st.columns(4)

    with c[0]:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )

    with c[1]:
        job = st.selectbox(
            "Job",
            [
                "admin.", "blue-collar", "entrepreneur",
                "housemaid", "management", "retired",
                "self-employed", "services", "student",
                "technician", "unemployed", "unknown"
            ]
        )

    with c[2]:
        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )

    with c[3]:
        education = st.selectbox(
            "Education",
            ["secondary", "tertiary", "primary", "unknown"]
        )

    st.divider()

    # FINANCIAL DETAILS
    st.subheader("💰 Financial Details")

    c = st.columns(4)

    with c[0]:
        balance = st.number_input(
            "Average Yearly Balance", value=1000
        )

    with c[1]:
        default = st.selectbox(
            "Credit in Default", ["no", "yes"]
        )

    with c[2]:
        housing = st.selectbox(
            "Housing Loan", ["yes", "no"]
        )

    with c[3]:
        loan = st.selectbox(
            "Personal Loan", ["no", "yes"]
        )

    st.divider()

    # CAMPAIGN DETAILS
    st.subheader("📞 Campaign Details")

    c = st.columns(8)

    with c[0]:
        contact = st.selectbox(
            "Contact Type",
            ["cellular", "telephone", "unknown"]
        )

    with c[1]:
        day = st.number_input(
            "Contact Day",
            min_value=1, max_value=31, value=15
        )

    with c[2]:
        month = st.selectbox(
            "Contact Month",
            [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ]
        )

    with c[3]:
        duration = st.number_input(
            "Call Duration (sec)",
            min_value=0, value=180
        )

    with c[4]:
        campaign = st.number_input(
            "Campaign Contacts",
            min_value=1, value=1
        )

    with c[5]:
        pdays = st.number_input(
            "Previous Contact Days",
            min_value=-1, value=-1
        )

    with c[6]:
        previous = st.number_input(
            "Previous Contacts",
            min_value=0, value=0
        )

    with c[7]:
        poutcome = st.selectbox(
            "Previous Outcome",
            ["unknown", "failure", "other", "success"]
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Subscription",
        use_container_width=True
    )


# PREDICTION
if submitted:

    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    try:
        # Apply the fitted preprocessing pipeline
        customer_processed = preprocessor.transform(customer)

        # Predict the class
        prediction = model.predict(customer_processed)[0]

        # Calculate probability of subscription
        probabilities = model.predict_proba(customer_processed)[0]
        class_index = list(model.classes_).index(1)
        probability = probabilities[class_index] * 100

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "The model predicts that the customer "
                "is likely to subscribe."
            )
        else:
            st.warning(
                "The model predicts that the customer "
                "is unlikely to subscribe."
            )

        st.metric(
            "Estimated Subscription Probability",
            f"{probability:.2f}%"
        )

        st.caption(
            "Predictions are estimates and do not guarantee "
            "actual customer behavior."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
