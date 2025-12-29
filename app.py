import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# 1. Load the Model
try:
    model = joblib.load('decision_tree_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please upload 'decision_tree_model.joblib'.")
    st.stop()

# 2. Page Config
st.set_page_config(page_title="Lung Health Dashboard", layout="wide")

# 3. Sidebar Inputs with Image
# Tries to load the image from the local folder. 
# If you haven't uploaded it yet, it just skips it (no error).
try:
    st.sidebar.image("lung_image.png", width=200)
except:
    st.sidebar.write("⚠️ Please upload 'lung_image.png' to GitHub")

st.sidebar.header("Patient Profile")
st.sidebar.write("Configure symptoms:")

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

# Prepare input vector
input_data = np.array([[allergy, swallowing, alcohol, coughing, fingers, chest_pain]])

# 4. Main Dashboard Area
st.title("🩺 Lung Health Risk Predictor")
st.markdown("### AI-Powered Symptom Analysis")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Prediction Analysis")
    
    if st.button("Run Diagnosis", type="primary"):
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
        name='Patient Data',
        line_color='#3498db'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=500,
        margin=dict(t=40, b=40, l=40, r=40)
    )

    st.plotly_chart(fig_radar, use_container_width=True)