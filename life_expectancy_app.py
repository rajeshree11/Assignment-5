import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Life Expectancy Prediction App", layout="centered")
st.title("🌍 Life Expectancy Prediction App")
st.markdown("Predict life expectancy using economic and demographic indicators.")

# 🚀 File Upload (Optional)
st.subheader("📂 Upload Dataset (Optional)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

# 📥 Manual Input
st.subheader("📝 Input Country Data")
gdp = st.number_input("GDP per Capita", value=3000.0)
population = st.number_input("Population", value=10_000_000.0)
year = st.number_input("Year", value=2007)
continent = st.selectbox("Continent", ["Asia", "Africa", "Europe", "North America", "South America", "Oceania"])

# 🔍 Load Model & Scaler
try:
    model = joblib.load("life_model.pkl")
    scaler = joblib.load("scaler.pkl")
    scaled_input = scaler.transform([[gdp, population, year]])
    st.success("Scaler loaded successfully.")
except:
    scaled_input = [[gdp, population, year]]
    st.warning("⚠️ Scaler not available. Raw input will be used.")

# 📊 Predict
if st.button("🧑‍⚕️ Predict Life Expectancy"):
    prediction = model.predict(scaled_input)
    st.success(f"📈 Predicted Life Expectancy: **{round(prediction[0], 2)} years**")
