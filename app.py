import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧")
st.title("🔧 Machine Failure Prediction")

@st.cache_resource
def load_model():
    return joblib.load("gradient_boosting_fe_threshold.pkl")

saved = load_model()
model, threshold, features = saved["model"], saved["threshold"], saved["features"]

machine_type = st.selectbox("Type", ["L", "M", "H"])
air = st.number_input("Air temperature [K]", 295.0, 305.0, 300.1, 0.1)
process = st.number_input("Process temperature [K]", 305.0, 314.0, 310.1, 0.1)
rpm = st.number_input("Rotational speed [rpm]", 1000.0, 3000.0, 1503.0, 1.0)
torque = st.number_input("Torque [Nm]", 3.0, 77.0, 40.1, 0.1)
wear = st.number_input("Tool wear [min]", 0.0, 260.0, 108.0, 1.0)

if st.button("Predict"):
    row = {
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Temp diff": process - air,
        "Power": torque * rpm * 2 * 3.14159 / 60,
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M"),
    }
    X = pd.DataFrame([row])[features]
    proba = model.predict_proba(X)[0, 1]

    st.metric("Failure probability", f"{proba:.1%}")
    st.progress(float(proba))

    if proba >= threshold:
        st.error("⚠️ الماكينة مرجح تتعطل، يفضل صيانة")
    else:
        st.success("✅ الماكينة سليمة")