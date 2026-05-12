import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn

# Page config
st.set_page_config(page_title="Paddy Yield Predictor", layout="wide")

# Load trained pipeline
@st.cache_data
def load_model():
    with open("Paddyield.pkl", "rb") as f:
        model = pickle.load(f)
    st.info(f"✅ Loaded model: {type(model).__name__}")
    st.info(f"Sklearn version: {sklearn.__version__}")
    return model

# Load model
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Paddyield.pkl not found! Please place the model file in the same directory.")
    st.stop()

st.title("🌾 Paddy Yield Prediction Model")
st.markdown("Enter agricultural parameters to predict paddy yield.")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Farm & Variety")
    Agriblock = st.number_input("Agriblock", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    Variety = st.number_input("Variety", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    Soil_Types = st.number_input("Soil Types", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    Nursery = st.number_input("Nursery", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    
    st.subheader("💧 Weather (Days 1-60)")
    Max_temp_D1_D30 = st.number_input("Max temp D1-D30 (°C)", min_value=-10.0, max_value=50.0, value=30.0, step=0.1)
    Min_temp_D31_D60 = st.number_input("Min temp D31-D60 (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    
    Relative_Humidity_D1_D30 = st.number_input("Relative Humidity D1-D30 (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    Relative_Humidity_D31_D60 = st.number_input("Relative Humidity D31-D60 (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

with col2:
    st.subheader("🌱 Crop Management")
    LP_nurseryarea_in_Tonnes = st.number_input("LP nursery area (Tonnes)", min_value=0.0, value=0.5, step=0.01, format="%.3f")
    Weed28D_thiobencarb = st.number_input("Weed28D thiobencarb (kg)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    Pest_60Day_in_ml = st.number_input("Pest 60Day (ml)", min_value=0.0, value=500.0, step=50.0, format="%.0f")
    
    st.subheader("🌤️ Weather (Days 61-120)")
    Min_temp_D61_D90 = st.number_input("Min temp D61-D90 (°C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.1)
    Max_temp_D61_D90 = st.number_input("Max temp D61-D90 (°C)", min_value=-10.0, max_value=50.0, value=32.0, step=0.1)
    Min_temp_D91_D120 = st.number_input("Min temp D91-D120 (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
    
    st.subheader("🌧️ Rainfall & Irrigation")
    Rain51_70_in_mm = st.number_input("Rain 51-70D (mm)", min_value=0.0, value=100.0, step=10.0)
    AI51_70_in_mm = st.number_input("AI 51-70D (mm)", min_value=0.0, value=50.0, step=10.0)
    AI71_105_in_mm = st.number_input("AI 71-105D (mm)", min_value=0.0, value=80.0, step=10.0)

# Additional parameters in new row
col3, col4, col5 = st.columns(3)
with col3:
    Inst_Wind_Speed_D31_D60_in_Knots = st.number_input("Wind Speed D31-D60 (knots)", min_value=0.0, value=5.0, step=0.5)
    Inst_Wind_Speed_D91_D120_in_Knots = st.number_input("Wind Speed D91-D120 (knots)", min_value=0.0, value=4.0, step=0.5)
    
with col4:
    Wind_Direction_D61_D90 = st.number_input("Wind Direction D61-D90 (°)", min_value=0.0, max_value=360.0, value=180.0, step=10.0)
    Wind_Direction_D91_D120 = st.number_input("Wind Direction D91-D120 (°)", min_value=0.0, max_value=360.0, value=200.0, step=10.0)
    
with col5:
    Relative_Humidity_D91_D120 = st.number_input("Relative Humidity D91-D120 (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

# Predict button
if st.button("🚀 Predict Paddy Yield", type="primary", use_container_width=True):
    try:
        # Create input dictionary matching your model training columns exactly
        sample_dict = {
            "Agriblock": Agriblock,
            "Variety": Variety,
            "Soil Types": Soil_Types,
            "Nursery": Nursery,
            "LP_nurseryarea(in Tonnes)": LP_nurseryarea_in_Tonnes,
            "Weed28D_thiobencarb": Weed28D_thiobencarb,
            "Pest_60Day(in ml)": Pest_60Day_in_ml,
            "51_70DRain(in mm)": Rain51_70_in_mm,
            "51_70AI(in mm)": AI51_70_in_mm,
            "71_105DAI(in mm)": AI71_105_in_mm,
            "Max temp_D1_D30": Max_temp_D1_D30,
            "Min temp_D31_D60": Min_temp_D31_D60,
            "Min temp_D61_D90": Min_temp_D61_D90,
            "Max temp_D61_D90": Max_temp_D61_D90,
            "Min temp_D91_D120": Min_temp_D91_D120,
            "Inst Wind Speed_D31_D60(in Knots)": Inst_Wind_Speed_D31_D60_in_Knots,
            "Inst Wind Speed_D91_D120(in Knots)": Inst_Wind_Speed_D91_D120_in_Knots,
            "Wind Direction_D61_D90": Wind_Direction_D61_D90,
            "Wind Direction_D91_D120": Wind_Direction_D91_D120,
            "Relative Humidity_D1_D30": Relative_Humidity_D1_D30,
            "Relative Humidity_D31_D60": Relative_Humidity_D31_D60,
            "Relative Humidity_D91_D120": Relative_Humidity_D91_D120
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([sample_dict])
        
        # Predict
        pred = model.predict(input_df)[0]
        predict = round(float(pred), 2)
        
        # Display results
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.success(f"**Predicted Paddy Yield: {predict}**")
            st.metric("Paddy Yield", f"{predict}", delta=None)
        with col_right:
            st.info("📈 Model Prediction")
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            st.dataframe(input_df.T, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.exception(e)

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    1. Fill in all agricultural parameters
    2. Click **Predict Paddy Yield**
    3. Review the predicted yield and input summary
    
    **Note**: Column names must match exactly with training data.
    """)

# Sidebar info
st.sidebar.title("📋 Model Info")
st.sidebar.info("Paddy Yield Prediction using ML Pipeline")
st.sidebar.markdown("**Required file:** `Paddyield.pkl`")