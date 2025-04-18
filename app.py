import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Custom CSS with tooltip range hover and UI enhancements
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
        padding: 30px;
    }
    .stButton>button {
        background-color: #ffffff;
        color: #374151;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #f3f4f6;
        transform: scale(1.05);
    }
    .default-button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .default-button:hover {
        background-color: #388E3C;
        transform: scale(1.05);
    }
    .stNumberInput input {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        padding: 10px;
        background-color: #fafafa;
        font-size: 16px;
    }
    .stNumberInput label {
        font-weight: bold;
        color: #374151;
        margin-bottom: 5px;
        font-size: 16px;
    }
    .stNumberInput div[role='spinbutton'] {
        border: 1px solid #d1d5db;
        border-radius: 4px;
        background-color: #e5e7eb;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .input-container {
        background-color: #fafafa;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        border: 1px solid #e5e7eb;
    }
    .result-box {
        border-radius: 8px;
        padding: 20px;
        margin-top: 30px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s;
    }
    .high-risk {
        background-color: #fee2e2;
        color: #dc2626;
    }
    .mid-risk {
        background-color: #fef9c3;
        color: #ca8a04;
    }
    .low-risk {
        background-color: #dcfce7;
        color: #15803d;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1, h2 {
        color: #1f2937;
        font-size: 28px;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        width: 18px;
        height: 18px;
        background-color: #e5e7eb;
        color: #6b7280;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        margin-left: 8px;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 180px;
        background-color: #374151;
        color: #ffffff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        top: -5px;
        left: 110%;
        font-size: 14px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .chat-icon {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #4CAF50;
        color: white;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    .chat-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 30px;
        border: 1px solid #e5e7eb;
    }
    .chat-box textarea {
        width: 100%;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 12px;
        font-size: 16px;
        background-color: #f0f7ff;
    }
    .chat-box button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        cursor: pointer;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'xgb_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# Define feature columns and risk mapping
features = scaler.feature_names_in_.tolist()
risk_mapping = {0: 'High Risk', 1: 'Mid Risk', 2: 'Low Risk'}

# Sidebar
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app predicts maternal health risk levels based on 8 clinical measurements:
    
    - **High Risk (0)**: Red - Urgent attention needed
    - **Mid Risk (1)**: Yellow - Monitor and consult
    - **Low Risk (2)**: Green - All good!
    
    **Features**:
    - Age
    - Body Temperature
    - Heart Rate
    - Systolic BP
    - Diastolic BP
    - BMI
    - HbA1c
    - Fasting Glucose
    
    Adjustable ranges allow flexibility beyond typical values.
    """)

# Main content
st.header("Maternal Risk Prediction")
st.markdown("Enter your health metrics to assess your risk level")

# Default test cases
default_cases = {
    'High Risk': {
        'Age': 20.0,
        'Body Temperature(F)': 97.5,
        'Heart rate(bpm)': 91.0,
        'Systolic Blood Pressure(mm Hg)': 161.0,
        'Diastolic Blood Pressure(mm Hg)': 100.0,
        'BMI(kg/m 2)': 24.9,
        'Blood Glucose(HbA1c)': 7.28,
        'Blood Glucose(Fasting hour-mg/dl)': 5.8
    },
    'Mid Risk': {
        'Age': 29.0,
        'Body Temperature(F)': 98.6,
        'Heart rate(bpm)': 84.0,
        'Systolic Blood Pressure(mm Hg)': 129.0,
        'Diastolic Blood Pressure(mm Hg)': 87.0,
        'BMI(kg/m 2)': 19.0,
        'Blood Glucose(HbA1c)': 7.67,
        'Blood Glucose(Fasting hour-mg/dl)': 6.4
    },
    'Low Risk': {
        'Age': 28.0,
        'Body Temperature(F)': 98.6,
        'Heart rate(bpm)': 79.0,
        'Systolic Blood Pressure(mm Hg)': 136.0,
        'Diastolic Blood Pressure(mm Hg)': 87.0,
        'BMI(kg/m 2)': 23.7,
        'Blood Glucose(HbA1c)': 4.56,
        'Blood Glucose(Fasting hour-mg/dl)': 4.4
    }
}

# Initialize session state
if 'input_values' not in st.session_state:
    st.session_state.input_values = default_cases['Low Risk'].copy()
if 'manual_mode' not in st.session_state:
    st.session_state.manual_mode = False
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'message_sent' not in st.session_state:
    st.session_state.message_sent = False

# Function to update inputs
def update_inputs(case):
    st.session_state.input_values = default_cases[case].copy()
    st.session_state.manual_mode = False
    st.session_state.show_chat = False
    st.session_state.prediction = None
    st.session_state.message_sent = False

# Input mode selection
st.subheader("Select Input Mode")
input_mode = st.radio("Choose how to provide inputs:", ("Default Test Cases", "Manual Input"))
st.session_state.manual_mode = (input_mode == "Manual Input")

# Default test case buttons
if not st.session_state.manual_mode:
    st.subheader("Test Default Cases")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Test High Risk", key="high_risk", help="Load High Risk test case", type="primary"):
            update_inputs('High Risk')
    with col2:
        if st.button("Test Mid Risk", key="mid_risk", help="Load Mid Risk test case", type="primary"):
            update_inputs('Mid Risk')
    with col3:
        if st.button("Test Low Risk", key="low_risk", help="Load Low Risk test case", type="primary"):
            update_inputs('Low Risk')

# Input fields
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
ranges = {
    'Age': {'min': 15, 'max': 50, 'step': 1, 'tooltip': 'Range: 15–50 years'},
    'Body Temperature(F)': {'min': 93.0, 'max': 104.0, 'step': 0.1, 'tooltip': 'Range: 93–104 °F'},
    'Heart rate(bpm)': {'min': 45, 'max': 150, 'step': 1, 'tooltip': 'Range: 45–150 bpm'},
    'Systolic Blood Pressure(mm Hg)': {'min': 90, 'max': 169, 'step': 1, 'tooltip': 'Range: 90–169 mm Hg'},
    'Diastolic Blood Pressure(mm Hg)': {'min': 70, 'max': 100, 'step': 1, 'tooltip': 'Range: 70–100 mm Hg'},
    'BMI(kg/m 2)': {'min': 14.9, 'max': 27.9, 'step': 0.1, 'tooltip': 'Range: 14.9–27.9 kg/m²'},
    'Blood Glucose(HbA1c)': {'min': 3.0, 'max': 10.0, 'step': 0.1, 'tooltip': 'Range: 3.0–10.0 %'},
    'Blood Glucose(Fasting hour-mg/dl)': {'min': 3.5, 'max': 8.9, 'step': 0.1, 'tooltip': 'Range: 3.5–8.9 mg/dL'}
}

inputs = {}
for i, feature in enumerate(features):
    label = feature.replace('(mm Hg)', '(mm Hg)').replace('(kg/m 2)', '(kg/m²)')
    col = col1 if i % 2 == 0 else col2
    with col:
        st.markdown(f"{label}<span class='tooltip'>?<span class='tooltiptext'>{ranges[feature]['tooltip']}</span></span>", unsafe_allow_html=True)
        if st.session_state.manual_mode:
            inputs[feature] = st.number_input(
                "",
                min_value=float(ranges[feature]['min']),
                max_value=float(ranges[feature]['max']),
                value=float(st.session_state.input_values[feature]),
                step=float(ranges[feature]['step']),
                key=feature,
                label_visibility="collapsed"
            )
        else:
            inputs[feature] = st.session_state.input_values[feature]
            st.markdown(f"<div style='padding: 10px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: #f3f4f6; font-size: 16px;'>{inputs[feature]}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Update session state if manual inputs changed
if st.session_state.manual_mode:
    st.session_state.input_values = inputs

# Prepare input data
input_data = pd.DataFrame([st.session_state.input_values], columns=features)

# Predict
if st.button("Predict Risk"):
    with st.spinner("Predicting..."):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = np.max(probabilities) * 100
        st.session_state.prediction = prediction
        st.session_state.confidence = confidence
        st.session_state.show_chat = False
        st.session_state.message_sent = False
        
        st.subheader("Prediction Results")
        result = risk_mapping[prediction]
        risk_class = {0: 'high-risk', 1: 'mid-risk', 2: 'low-risk'}[prediction]
        recommendation = {
            0: 'Immediate medical attention recommended',
            1: 'Monitor and consult a healthcare provider',
            2: 'All good! Continue regular check-ups'
        }[prediction]
        
        st.markdown(f"""
        <div class='result-box {risk_class}'>
            <b>{result} (Prediction: {prediction})</b><br>
            Recommendation: {recommendation}<br>
            Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

# Floating chat icon
st.markdown('<div class="chat-icon" onclick="document.getElementById(\'chat_trigger\').click()">💬</div>', unsafe_allow_html=True)

# Hidden button to trigger chat
if st.button("Trigger Chat", key="chat_trigger", help="Click to open chat"):
    if st.session_state.prediction is not None:
        st.session_state.show_chat = True
    else:
        st.warning("Please make a prediction first by clicking 'Predict Risk'.")

# Chat box logic
if st.session_state.show_chat and st.session_state.prediction is not None:
    st.subheader("Chat with a Doctor")
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    values = st.session_state.input_values
    default_chat = (
        f"I have these values: Age={values['Age']}, Temp={values['Body Temperature(F)']}°F, "
        f"HR={values['Heart rate(bpm)']} bpm, BP={values['Systolic Blood Pressure(mm Hg)']}/{values['Diastolic Blood Pressure(mm Hg)']} mmHg, "
        f"BMI={values['BMI(kg/m 2)']} kg/m², HbA1c={values['Blood Glucose(HbA1c)']}, Fasting Glucose={values['Blood Glucose(Fasting hour-mg/dl)']} mg/dl. "
        f"Risk level predicted: {risk_mapping[st.session_state.prediction]}. What should I do?"
    )
    
    chat_input = st.text_area("Your message to the doctor:", value=default_chat, height=150)
    
    if st.button("Send to Doctor") and not st.session_state.message_sent:
        st.session_state.message_sent = True
    
    if st.session_state.message_sent:
        st.success("Message sent successfully!")
        st.markdown("Please wait for the doctor's response, or submit new values for another prediction.")
    
    st.markdown('</div>', unsafe_allow_html=True)
