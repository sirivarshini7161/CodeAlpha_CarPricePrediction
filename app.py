import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Load model and feature list
model = joblib.load("data/processed/car_price_model.pkl")
feature_names = pd.read_csv("data/processed/model_features.csv")['0'].tolist()

st.title("🚗 Car Price Prediction Dashboard")
st.markdown("Predict a used car's fair selling price based on its specifications, using a Random Forest model trained on real used car listings.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Car Details")
    present_price = st.number_input("Present Price (Lakhs ₹)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    year = st.number_input("Manufacturing Year", min_value=2000, max_value=2020, value=2015)
    driven_kms = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Number of Previous Owners", [0, 1, 3])

    predict_button = st.button("Predict Price", type="primary")

with col2:
    st.subheader("Prediction Result")

    if predict_button:
        car_age = 2020 - year

        input_dict = {col: 0 for col in feature_names}
        input_dict['Present_Price'] = present_price
        input_dict['Driven_kms'] = driven_kms
        input_dict['Owner'] = owner
        input_dict['Car_Age'] = car_age

        if fuel_type == "Diesel":
            input_dict['Fuel_Type_Diesel'] = 1
        elif fuel_type == "Petrol":
            input_dict['Fuel_Type_Petrol'] = 1

        if selling_type == "Individual":
            input_dict['Selling_type_Individual'] = 1

        if transmission == "Manual":
            input_dict['Transmission_Manual'] = 1

        input_df = pd.DataFrame([input_dict])[feature_names]
        prediction = model.predict(input_df)[0]

        st.metric("Predicted Selling Price", f"₹ {prediction:.2f} Lakhs")
        st.caption(f"Based on a car that is {car_age} years old with {driven_kms:,} km driven.")
    else:
        st.info("Fill in the car details and click 'Predict Price' to see the estimated selling price.")

st.markdown("---")
st.subheader("What Drives This Model's Predictions?")
st.image("visualizations/01_feature_importance.png", use_container_width=True)
st.caption("Present Price and Car Age are by far the strongest predictors of resale value in this model.")