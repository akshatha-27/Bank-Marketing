import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load trained model and fitted preprocessor
@st.cache_resource
def load_models():
    model = joblib.load("random_forest_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_models()
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop()

# Custom styling

st.markdown("""
<style>
/* Center the main heading */
h1 {
    text-align: center;
}

/* Center the description */
.stApp p {
    text-align: center;
}

/* Center the main content and limit its width */
.block-container {
    max-width: 1000px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 2rem;
}

/* Center and narrow the form */
div[data-testid="stForm"] {
    max-width: 850px;
    margin-left: auto;
    margin-right: auto;
}

/* Red prediction button */
div[data-testid="stFormSubmitButton"] button {
    background-color: #DC2626;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #B91C1C;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# Application heading
st.title("🏦 Bank Marketing Prediction")

st.write(
    "Predict whether a customer is likely to subscribe "
    "to a term deposit using a Random Forest classifier."
)

st.divider()

# Customer input form
with st.form("customer_form"):

    st.subheader("👤 Personal Details")

    age = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur",
            "housemaid", "management", "retired",
            "self-employed", "services", "student",
            "technician", "unemployed", "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"]
    )

    education = st.selectbox(
        "Education",
        ["secondary", "tertiary", "primary", "unknown"]
    )

    st.divider()

    st.subheader("💰 Financial Details")

    balance = st.number_input(
        "Average Yearly Balance", value=1000
    )

    default = st.selectbox(
        "Credit in Default", ["no", "yes"]
    )

    housing = st.selectbox(
        "Housing Loan", ["yes", "no"]
    )

    loan = st.selectbox(
        "Personal Loan", ["no", "yes"]
    )

    st.divider()

    st.subheader("📞 Campaign Details")

    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        "Contact Day", min_value=1, max_value=31, value=15
    )

    month = st.selectbox(
        "Contact Month",
        [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
    )

    duration = st.number_input(
        "Call Duration (seconds)", min_value=0, value=180
    )

    campaign = st.number_input(
        "Contacts During This Campaign",
        min_value=1, value=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=-1, value=-1
    )

    previous = st.number_input(
        "Previous Contacts", min_value=0, value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Subscription",
        use_container_width=True
    )

# Prediction section
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

        # Get probabilities for both classes
        probabilities = model.predict_proba(customer_processed)[0]

        # Find probability positions using the model's class labels
        classes = list(model.classes_)
        class_index_yes = classes.index(1)
        class_index_no = classes.index(0)

        subscription_probability = (
            probabilities[class_index_yes] * 100
        )

        non_subscription_probability = (
            probabilities[class_index_no] * 100
        )

        # Display results
        st.divider()
        st.subheader("📊 Prediction Result")

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

        # Display both probabilities
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="✅ Probability of Subscribing",
                value=f"{subscription_probability:.2f}%"
            )
            st.progress(
                min(max(subscription_probability / 100, 0.0), 1.0)
            )

        with col2:
            st.metric(
                label="❌ Probability of Not Subscribing",
                value=f"{non_subscription_probability:.2f}%"
            )
            st.progress(
                min(max(non_subscription_probability / 100, 0.0), 1.0)
            )

        st.caption(
            "Predictions are estimates and do not guarantee "
            "actual customer behavior."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
