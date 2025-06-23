# life_expectancy_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and optional scaler
model = joblib.load("life_expectancy_model.pkl")
try:
    scaler = joblib.load("scaler.pkl")
    scaler_available = True
except:
    scaler_available = False

# Page configuration
st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🌍 Life Expectancy Prediction")
st.markdown("Upload a dataset or enter details below to predict life expectancy.")

# File Upload
st.subheader("📤 Upload a CSV file (optional)")
uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])

# Define prediction function
def predict_life_expectancy(data):
    if scaler_available:
        data_scaled = scaler.transform(data)
        return model.predict(data_scaled)
    else:
        st.warning("⚠️ Scaler not available. Raw input will be used.")
        return model.predict(data)

# Manual Input Form
st.markdown("---")
st.subheader("📝 Manual Input for Prediction")

gdp = st.number_input("GDP per Capita", min_value=0.0, value=3000.0, step=100.0)
population = st.number_input("Population", min_value=1000.0, value=10000000.0, step=1000.0)
year = st.number_input("Year", min_value=2000, max_value=2025, value=2007)
continent = st.selectbox("Continent", ["Asia", "Europe", "Africa", "Americas", "Oceania"])

# Encode Continent
continent_map = {"Asia": 0, "Europe": 1, "Africa": 2, "Americas": 3, "Oceania": 4}
continent_encoded = continent_map.get(continent, 0)

# Predict button
if st.button("🧠 Predict Life Expectancy"):
    input_array = np.array([[gdp, population, year, continent_encoded]])
    prediction = predict_life_expectancy(input_array)
    st.success(f"📈 Predicted Life Expectancy: **{prediction[0]:.2f} years**")

# Handle CSV upload
if uploaded_file:
    st.markdown("---")
    st.subheader("🔍 Predictions from Uploaded Data")

    df = pd.read_csv(uploaded_file)
    st.write("📄 Preview of Uploaded Dataset", df.head())

    # Assume column names match exactly
    if set(["GDP", "Population", "Year", "Continent"]).issubset(df.columns):
        df["Continent_encoded"] = df["Continent"].map(continent_map)
        input_data = df[["GDP", "Population", "Year", "Continent_encoded"]]
        df["Predicted_Life_Expectancy"] = predict_life_expectancy(input_data)
        st.write("✅ Predictions", df)

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="life_expectancy_predictions.csv", mime="text/csv")
    else:
        st.error("❌ CSV must contain the columns: GDP, Population, Year, Continent")

# Expandable model info
with st.expander("ℹ️ About the Model"):
    st.write("""
    This app uses a machine learning model trained on World Health and Economic data 
    to predict life expectancy based on GDP, population, year, and continent.
    """)

