import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# 2. Page Config
st.set_page_config(page_title="Lung Health Dashboard", layout="wide")

# 1. Load the Model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    try:
        return joblib.load('decision_tree_model.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please upload 'decision_tree_model.joblib'.")
        st.stop()

model = load_model()

# Cache image loading
@st.cache_data
def load_image(path):
    return Image.open(path)

# 3. Sidebar Inputs with Image
# Tries to load the image from the local folder.
# If you haven't uploaded it yet, it just skips it (no error).
try:
    img = load_image("lung_image.png")
    st.sidebar.image(img, width=200)
except:
    st.sidebar.write("⚠️ Please upload 'lung_image.png' to GitHub")

st.sidebar.header("Patient Profile")
st.sidebar.write("Configure symptoms:")

# Model info in sidebar
st.sidebar.markdown("---")
st.sidebar.info("🤖 Model: Decision Tree Classifier\n\n⚠️ For informational purposes only. Not a substitute for medical advice.")

def create_toggle(label):
    val = st.sidebar.radio(label, ["No", "Yes"], horizontal=True, index=0)
    return 1 if val == "Yes" else 0

# Input fields
allergy = create_toggle("🤧 Allergy")
swallowing = create_toggle("🤐 Swallowing Difficulty")
alcohol = create_toggle("🍷 Alcohol Consuming")
coughing = create_toggle("😮‍💨 Coughing")
fingers = create_toggle("🖐️ Yellow Fingers")
chest_pain = create_toggle("💔 Chest Pain")

# Prepare input data as named DataFrame to prevent silent bugs with named features
feature_names = ['ALLERGY', 'SWALLOWING DIFFICULTY', 'ALCOHOL CONSUMING', 'COUGHING', 'YELLOW_FINGERS', 'CHEST PAIN']
input_data = pd.DataFrame([[allergy, swallowing, alcohol, coughing, fingers, chest_pain]], columns=feature_names)

# 4. Main Dashboard Area
st.title("🩺 Lung Health Risk Predictor")
st.markdown("### AI-Powered Symptom Analysis")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Prediction Analysis")
    
    if st.button("Run Diagnosis", type="primary"):
        with st.spinner("Analyzing symptoms..."):
            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]
        
        # Logic assuming 1 = High Risk
        class_label = "High Risk" if prediction == 1 or str(prediction).upper() == "YES" else "Low Risk"
        confidence = probs[1] if prediction == 1 else probs[0]
        
        if class_label == "High Risk":
            st.error(f"Prediction: {class_label}")
        else:
            st.success(f"Prediction: {class_label}")
            
        st.metric("Model Confidence", f"{confidence:.1%}")

        # Donut Chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Healthy', 'Risk'],
            values=[probs[0], probs[1]],
            hole=.6,
            marker_colors=['#2ecc71', '#e74c3c']
        )])
        fig_donut.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.subheader("Symptom Profile")
    
    # Radar Chart
    categories = ['Allergy', 'Swallowing', 'Alcohol', 'Coughing', 'Yellow Fingers', 'Chest Pain']
    values = [allergy, swallowing, alcohol, coughing, fingers, chest_pain]
    
    # Close the loop
    values += values[:1]
    categories += categories[:1]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='#3498db', width=2)
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=350,
        margin=dict(t=30, b=30, l=30, r=30)
    )

    st.plotly_chart(fig_radar, use_container_width=True)
