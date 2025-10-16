import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
from src.custom_objects import column_ratio, ratio_name, ClusterSimilarity, ratio_pipeline



# Paths
PROJECT_ROOT = Path().resolve()
MODEL_PATH = PROJECT_ROOT / "models" / "my_california_housing_model.pkl"
DATA_PATH = PROJECT_ROOT / "datasets" / "housing/housing.csv"


ocean_options = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

# Load model
model = joblib.load(MODEL_PATH)

# App title
st.title("California Housing Price Prediction")

# Load data
housing = pd.read_csv(DATA_PATH)

# Inputs
median_income = st.number_input("Median Income", float(housing["median_income"].min()), float(housing["median_income"].max()))
house_age = st.number_input("House Age", float(housing["housing_median_age"].min()), float(housing["housing_median_age"].max()))
rooms = st.number_input("Total Rooms", float(housing["total_rooms"].min()), float(housing["total_rooms"].max()))
bedrooms = st.number_input("Total Bedrooms", float(housing["total_bedrooms"].min()), float(housing["total_bedrooms"].max()))
population = st.number_input("Population", float(housing["population"].min()), float(housing["population"].max()))
households = st.number_input("Households", float(housing["households"].min()), float(housing["households"].max()))
longitude = st.number_input("Longitude", float(housing["longitude"].min()), float(housing["longitude"].max()))
latitude = st.number_input("Latitude", float(housing["latitude"].min()), float(housing["latitude"].max()))
ocean_proximity = st.selectbox("Ocean Proximity", options=ocean_options)

# Prediction
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1f77b4;
    color:white;
    height:3em;
    width:12em;
    font-size:18px;
    border-radius: 5px;
}
div.stButton > button:first-child:hover {
    background-color: #0b5394;
    color:white;
}
</style>""", unsafe_allow_html=True)

if st.button("Predict Price"):
    # Your prediction logic here
    input_df = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [house_age],
    "total_rooms": [rooms],
    "total_bedrooms": [bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})
    price = model.predict(input_df)
    st.success(f"Predicted Median House Value: ${price[0]:,.2f}")


