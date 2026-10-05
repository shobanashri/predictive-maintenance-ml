import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "predictive_maintenance_model.pkl"
)


@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact["threshold"]


pipeline, threshold = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⚙️ Predictive Maintenance System")

st.write(
    "Predict the probability of machine failure "
    "using machine operating conditions."
)

st.divider()

# --------------------------------------------------
# MACHINE INPUTS
# --------------------------------------------------

st.subheader("Machine Information")

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=290.0,
        max_value=310.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=300.0,
        max_value=320.0,
        value=310.0,
        step=0.1
    )


with col2:

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

st.divider()

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

predict_button = st.button(
    "🔍 Predict Machine Failure",
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    probability = pipeline.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= threshold
    )

    probability_percentage = probability * 100

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Failure Probability",
        f"{probability_percentage:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ Machine Failure Risk Detected"
        )

        st.write(
            "The current operating conditions indicate "
            "a higher probability of machine failure. "
            "Preventive maintenance should be considered."
        )

    else:

        st.success(
            "✅ No Failure Risk Detected"
        )

        st.write(
            "The current operating conditions indicate "
            "a lower probability of machine failure."
        )

    st.caption(
        f"Decision threshold: {threshold:.2f}"
    )