
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessor
classifier = joblib.load("churn_model.pkl")

ct = joblib.load("preprocessor.joblib")

# App title
st.title("Customer Churn Predictor")
st.write("Enter customer details to estimate the chance they will leave.")

# Customer details
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior citizen?", [0, 1])
partner = st.selectbox("Has a partner?", ["Yes", "No"])
dependents = st.selectbox("Has dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone = st.selectbox("Phone service?", ["Yes", "No"])
multiple = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless billing?", ["Yes", "No"])
payment = st.selectbox(
    "Payment method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
monthly = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0)
total = st.number_input("Total charges ($)", min_value=0.0, value=840.0)

if st.button("Predict Churn"):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }]
    )
# Create the TenureGroup feature using the same bins as training
    customer["TenureGroup"] = pd.cut( 
    customer["tenure"], 
     bins=[0, 12, 24, 36, 48, 60, 72]
       )



     
    # Apply the same preprocessing used during training
    customer_encoded = ct.transform(customer)

    # Predict churn probability
    probability = classifier.predict_proba(customer_encoded)[0][1]

    st.metric("Churn probability", f"{probability:.1%}")

    if probability >= 0.5:
        st.error("High risk of leaving")
    else:
        st.success("Likely to stay")