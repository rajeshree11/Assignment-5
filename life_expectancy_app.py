import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# 🔧 Define the model architecture
class LifeExpectancyModel(nn.Module):
    def __init__(self):
        super(LifeExpectancyModel, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 🧠 Load trained model
model = LifeExpectancyModel()
MODEL_PATH = "life_expectancy_model.pth"

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")

# 🧮 Scaler values (replace with your own if needed)
scaler_mean = np.array([4532.5, 13700000, 2007, 0.2, 0.2, 0.2, 0.2, 0.2])
scaler_scale = np.array([2100.0, 6800000, 10.0, 0.4, 0.4, 0.4, 0.4, 0.4])

# 🌍 App Layout
st.title("🌍 Life Expectancy Prediction")
st.markdown("Upload a dataset or enter details below to predict life expectancy.")

# 📁 File Upload
uploaded_file = st.file_uploader("📤 Upload a CSV file (optional)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("📊 **Preview of uploaded dataset:**")
    st.dataframe(df.head())

st.divider()
st.header("📝 Manual Input for Prediction")

# 🧾 Input Features
gdp = st.number_input("GDP per Capita", min_value=0.0, value=3000.0)
population = st.number_input("Population", min_value=0.0, value=10000000.0)
year = st.number_input("Year", min_value=1950, max_value=2025, value=2007)

# 🌎 Continent
continent = st.selectbox("Continent", ["Asia", "Europe", "Africa", "Americas", "Oceania"])
continent_map = {
    "Asia":     [1, 0, 0, 0, 0],
    "Europe":   [0, 1, 0, 0, 0],
    "Africa":   [0, 0, 1, 0, 0],
    "Americas": [0, 0, 0, 1, 0],
    "Oceania":  [0, 0, 0, 0, 1]
}

# 🧠 Prediction
if st.button("🔮 Predict Life Expectancy"):
    input_vector = [gdp, population, year] + continent_map[continent]
    input_array = np.array([input_vector], dtype=np.float32)
    input_scaled = (input_array - scaler_mean) / scaler_scale
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    st.success(f"📈 Predicted Life Expectancy: **{prediction:.2f} years**")

