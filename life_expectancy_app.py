import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 🎯 Model Architecture
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

# 🚀 Initialize Model
model = LifeExpectancyModel()
MODEL_PATH = "life_expectancy_model.pth"

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")

# ✳️ Hardcoded Scaler (replace values with your real means/stds if known)
scaler_mean = np.array([4532.5, 13700000, 2007, 0.2, 0.2, 0.2, 0.2, 0.2])
scaler_scale = np.array([2100.0, 6800000, 10.0, 0.4, 0.4, 0.4, 0.4, 0.4])

# 🌍 App UI
st.title("🌍 Life Expectancy Prediction App")
st.markdown("Predict life expectancy using economic and demographic indicators.")

# 📋 Inputs
st.header("📋 Input Country Data")
gdp = st.number_input("GDP per Capita", min_value=0.0, value=3000.0)
population = st.number_input("Population", min_value=0.0, value=10000000.0)
year = st.number_input("Year", min_value=1950, max_value=2025, value=2007)
continent = st.selectbox("Continent", ["Asia", "Europe", "Africa", "Americas", "Oceania"])

# 🌐 Continent One-Hot Encoding
continent_map = {
    "Asia":     [1, 0, 0, 0, 0],
    "Europe":   [0, 1, 0, 0, 0],
    "Africa":   [0, 0, 1, 0, 0],
    "Americas": [0, 0, 0, 1, 0],
    "Oceania":  [0, 0, 0, 0, 1]
}

input_vector = [gdp, population, year] + continent_map[continent]
input_array = np.array([input_vector], dtype=np.float32)

# 🧪 Normalize
input_scaled = (input_array - scaler_mean) / scaler_scale
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# 🔮 Predict
if st.button("🔮 Predict Life Expectancy"):
    with torch.no_grad():
        prediction = model(input_tensor).item()
    st.success(f"📈 Predicted Life Expectancy: **{prediction:.2f} years**")
