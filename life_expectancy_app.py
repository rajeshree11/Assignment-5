import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

# Define the model architecture (8 input features)
class LifeExpectancyModel(nn.Module):
    def __init__(self):
        super(LifeExpectancyModel, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # 8 input features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = LifeExpectancyModel()

# Load model weights
MODEL_PATH = "life_expectancy_model.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    st.sidebar.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.sidebar.error("❌ Model file not found!")

# Load scaler
SCALER_PATH = "scaler.pkl"
try:
    scaler = joblib.load(SCALER_PATH)
    st.sidebar.success("✅ Scaler loaded successfully")
except FileNotFoundError:
    st.sidebar.error("❌ Scaler file not found!")

# App title
st.title("🌍 Life Expectancy Prediction App")
st.markdown("Predict life expectancy using economic and demographic indicators.")

# Manual input form
st.header("📋 Input Country Data")
gdp = st.number_input("GDP per Capita", min_value=0.0, value=3000.0)
population = st.number_input("Population", min_value=0.0, value=10000000.0)
year = st.number_input("Year", min_value=1950, max_value=2025, value=2007)

# One-hot encode continent
continent = st.selectbox("Continent", ["Asia", "Europe", "Africa", "Americas", "Oceania"])
continent_encoding = {
    "Asia":     [1, 0, 0, 0, 0],
    "Europe":   [0, 1, 0, 0, 0],
    "Africa":   [0, 0, 1, 0, 0],
    "Americas": [0, 0, 0, 1, 0],
    "Oceania":  [0, 0, 0, 0, 1]
}

# Create input vector
input_features = [gdp, population, year] + continent_encoding[continent]
input_array = np.array([input_features], dtype=np.float32)

# Normalize input
if 'scaler' in locals():
    input_scaled = scaler.transform(input_array)
else:
    st.warning("⚠️ Scaler not available. Raw input will be used.")
    input_scaled = input_array

input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Predict
if st.button("🔮 Predict Life Expectancy"):
    with torch.no_grad():
        prediction = model(input_tensor).item()
    st.success(f"📈 Predicted Life Expectancy: **{prediction:.2f} years**")
