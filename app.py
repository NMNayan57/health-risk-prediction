import streamlit as st
import pickle
import numpy as np

# Custom CSS for improved UI and floating chat icon
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .recommendation {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    /* Floating chat icon */
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
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        z-index: 1000;
    }
    /* Chat box styling */
    .chat-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-top: 20px;
        border: 1px solid #e0e0e0;
    }
    .chat-box textarea {
        width: 100%;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
        background-color: #f0f7ff;
    }
    .chat-box button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit app
st.title("Advanced Health Risk Prediction")
st.markdown("Enter your health metrics below to assess your risk level", unsafe_allow_html=True)

# Initialize session state for chat visibility and prediction results
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'input_values' not in st.session_state:
    st.session_state.input_values = None
if 'message_sent' not in st.session_state:
    st.session_state.message_sent = False

# Input form with two columns
with st.form(key='health_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=130, value=26, step=1, help="Range: 15-120, adjustable ±10")
        body_temp = st.number_input("Body Temperature (F)", min_value=85.0, max_value=110.0, value=98.6, step=0.1, help="Range: 90-104, adjustable ±5")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=160, value=86, step=1, help="Range: 45-150, adjustable ±10")
        systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=80, max_value=180, value=129, step=1, help="Range: 90-169, adjustable ±10")
    
    with col2:
        diastolic_bp = st.number_input("Diastolic BP (mm Hg)", min_value=25, max_value=150, value=87, step=1, help="Range: 30-142, adjustable ±10")
        bmi = st.number_input("BMI (kg/m²)", min_value=13.0, max_value=30.0, value=21.4, step=0.1, help="Range: 14.9-27.9, adjustable ±2")
        hba1c = st.number_input("Blood Glucose (HbA1c)", min_value=3.0, max_value=9.5, value=5.5, step=0.1, help="Range: 3.5-8.9, adjustable ±0.5")
        fasting_glucose = st.number_input("Fasting Glucose (mg/dl)", min_value=45, max_value=165, value=100, step=1, help="Range: 50-150, adjustable ±15")

    # Submit button
    submit_button = st.form_submit_button(label='Predict Risk')

# Process prediction
if submit_button:
    # Create input array
    input_data = np.array([[age, body_temp, heart_rate, systolic_bp, diastolic_bp, bmi, hba1c, fasting_glucose]])
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence = max(probability)
    
    # Store results in session state
    st.session_state.prediction = prediction
    st.session_state.confidence = confidence
    st.session_state.input_values = {
        'age': age, 'body_temp': body_temp, 'heart_rate': heart_rate,
        'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'bmi': bmi, 'hba1c': hba1c, 'fasting_glucose': fasting_glucose
    }
    st.session_state.show_chat = False  # Reset chat visibility
    st.session_state.message_sent = False  # Reset message sent status
    
    # Display results with enhanced styling
    st.subheader("Prediction Results")
    
    if prediction == 0:
        st.markdown('<div class="recommendation" style="background-color: #ffebee;"><b>High Risk</b> (Prediction: 0)<br>Recommendation: Immediate medical attention recommended</div>', unsafe_allow_html=True)
    elif prediction == 2:
        st.markdown('<div class="recommendation" style="background-color: #fff3e0;"><b>Medium Risk</b> (Prediction: 2)<br>Recommendation: Monitor closely and consult healthcare provider</div>', unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown('<div class="recommendation" style="background-color: #e8f5e9;"><b>Low Risk</b> (Prediction: 1)<br>Recommendation: Maintain healthy lifestyle</div>', unsafe_allow_html=True)
    else:
        st.write(f"Unexpected prediction value: {prediction}")
    
    st.write(f"Confidence: {confidence:.2%}")

# Floating chat icon
st.markdown('<div class="chat-icon" onclick="document.getElementById(\'chat_trigger\').click()">💬</div>', unsafe_allow_html=True)

# Hidden button to trigger chat visibility
if st.button("Trigger Chat", key="chat_trigger", help="Click to open chat", type="primary"):
    st.session_state.show_chat = True

# Chat box logic
if st.session_state.show_chat and st.session_state.prediction is not None:
    st.subheader("Chat with a Doctor")
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    
    # Pre-fill chat input with current values
    values = st.session_state.input_values
    default_chat = (
        f"I have these values: Age={values['age']}, Temp={values['body_temp']}°F, "
        f"HR={values['heart_rate']} bpm, BP={values['systolic_bp']}/{values['diastolic_bp']} mmHg, "
        f"BMI={values['bmi']} kg/m², HbA1c={values['hba1c']}, Fasting Glucose={values['fasting_glucose']} mg/dl. "
        f"Risk level predicted: {'High' if st.session_state.prediction == 0 else 'Medium' if st.session_state.prediction == 2 else 'Low'}. "
        f"What should I do?"
    )
    
    chat_input = st.text_area("Your message to the doctor:", value=default_chat, height=150)
    
    if st.button("Send to Doctor") and not st.session_state.message_sent:
        st.session_state.message_sent = True
    
    if st.session_state.message_sent:
        st.success("Message sent successfully!")
        st.write("Please wait for the doctor's response, or [click here](#) to submit new values for another prediction.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app predicts health risk levels based on 8 clinical measurements:
    - **High Risk (0)**: Red - Urgent attention needed
    - **Medium Risk (2)**: Yellow - Monitor and consult
    - **Low Risk (1)**: Green - All good!
    
    **Features:**
    - Age
    - Body Temperature
    - Heart Rate
    - Systolic BP
    - Diastolic BP
    - BMI
    - HbA1c
    - Fasting Glucose
    """)
    st.markdown("Adjustable ranges allow flexibility beyond typical values.")
