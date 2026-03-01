import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# Page Config
st.set_page_config(page_title="Lung Health Dashboard", layout="wide")

# Load the Model (cached to avoid reloading on every interaction)
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

# --- Sidebar ---
try:
    img = load_image("lung_image.png")
    st.sidebar.image(img, width=200)
except:
    st.sidebar.write("⚠️ Please upload 'lung_image.png' to GitHub")

st.sidebar.header("Patient Profile")
st.sidebar.write("Toggle each symptom below — the dashboard updates instantly.")

def create_toggle(label):
    val = st.sidebar.radio(label, ["No", "Yes"], horizontal=True, index=0)
    return 1 if val == "Yes" else 0

# Symptom inputs
allergy      = create_toggle("🤧 Allergy")
swallowing   = create_toggle("🤐 Swallowing Difficulty")
alcohol      = create_toggle("🍷 Alcohol Consuming")
coughing     = create_toggle("😮‍💨 Coughing")
fingers      = create_toggle("🖐️ Yellow Fingers")
chest_pain   = create_toggle("💔 Chest Pain")

# Active symptom count shown live in sidebar
active_count = allergy + swallowing + alcohol + coughing + fingers + chest_pain
st.sidebar.markdown("---")
st.sidebar.metric("Active Symptoms", f"{active_count} / 6")

st.sidebar.markdown("---")
st.sidebar.info("🤖 Model: Decision Tree Classifier\n\n⚠️ For informational purposes only. Not a substitute for medical advice.")

# --- Input DataFrame ---
# Note: 'ALLERGY ' has a trailing space — this matches the feature name stored in the trained model.
feature_names = ['ALLERGY ', 'SWALLOWING DIFFICULTY', 'ALCOHOL CONSUMING', 'COUGHING', 'YELLOW_FINGERS', 'CHEST PAIN']
input_data = pd.DataFrame([[allergy, swallowing, alcohol, coughing, fingers, chest_pain]], columns=feature_names)

# --- Live Prediction (no button needed) ---
with st.spinner("Analyzing symptoms..."):
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

class_label = "High Risk" if prediction == 1 or str(prediction).upper() == "YES" else "Low Risk"
confidence  = probs[1] if prediction == 1 else probs[0]

# Map symptom names to their values for badges and reports
symptom_map = {
    "🤧 Allergy": allergy,
    "🤐 Swallowing Difficulty": swallowing,
    "🍷 Alcohol": alcohol,
    "😮‍💨 Coughing": coughing,
    "🖐️ Yellow Fingers": fingers,
    "💔 Chest Pain": chest_pain,
}

# --- Main Dashboard ---
st.title("🩺 Lung Health Risk Predictor")
st.markdown("### AI-Powered Symptom Analysis")

# Coloured symptom badges — red when active, grey when inactive
badge_html = " ".join([
    f'<span style="background:{"#e74c3c" if v else "#95a5a6"};color:white;'
    f'padding:4px 12px;border-radius:12px;margin:2px;font-size:0.85em">{k}</span>'
    for k, v in symptom_map.items()
])
st.markdown(badge_html, unsafe_allow_html=True)
st.markdown("")

# --- Tabs ---
tab_diagnosis, tab_history, tab_about = st.tabs(["📊 Diagnosis", "📋 History", "ℹ️ About"])

# ── Diagnosis Tab ──────────────────────────────────────────────────────────────
with tab_diagnosis:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Prediction Analysis")

        if class_label == "High Risk":
            st.error(f"🔴 Prediction: {class_label}")
        else:
            st.success(f"🟢 Prediction: {class_label}")

        m1, m2 = st.columns(2)
        m1.metric("Model Confidence", f"{confidence:.1%}")
        m2.metric("Active Symptoms", f"{active_count} / 6")

        # Donut chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Healthy', 'Risk'],
            values=[probs[0], probs[1]],
            hole=.6,
            marker_colors=['#2ecc71', '#e74c3c']
        )])
        fig_donut.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

        # Downloadable plain-text report
        report_lines = (
            ["Lung Health Risk Report", "=" * 26,
             f"Prediction: {class_label}",
             f"Confidence: {confidence:.1%}",
             f"Active Symptoms: {active_count}/6", "",
             "Symptom Details:"]
            + [f"  {k}: {'Yes' if v else 'No'}" for k, v in symptom_map.items()]
        )
        st.download_button(
            "📥 Download Report",
            data="\n".join(report_lines),
            file_name="lung_risk_report.txt",
            mime="text/plain",
        )

    with col2:
        st.subheader("Symptom Profile")

        categories = ['Allergy', 'Swallowing', 'Alcohol', 'Coughing', 'Yellow Fingers', 'Chest Pain']
        values = [allergy, swallowing, alcohol, coughing, fingers, chest_pain]
        # Close the radar loop
        values     += values[:1]
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
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=350,
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── History Tab ────────────────────────────────────────────────────────────────
with tab_history:
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("💾 Save Current Result to History"):
        st.session_state.history.append({
            "Prediction": class_label,
            "Confidence": f"{confidence:.1%}",
            "Active Symptoms": active_count,
            **{k: "Yes" if v else "No" for k, v in symptom_map.items()},
        })
        st.success("Result saved!")

    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No saved results yet. Toggle symptoms, then click **Save Current Result to History**.")

# ── About Tab ──────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## About This App

This dashboard uses a **Decision Tree Classifier** trained on lung cancer symptom data
to estimate risk based on 6 key symptoms.

### How to Use
1. Toggle your symptoms in the **sidebar** on the left.
2. The prediction and charts update **instantly** — no button required.
3. Use the **📊 Diagnosis** tab to view your risk score and charts.
4. Use the **📋 History** tab to save and compare multiple readings.
5. Download a plain-text summary with the **📥 Download Report** button.

### Symptoms Assessed
| Symptom | Description |
|---|---|
| 🤧 Allergy | Persistent allergic reactions |
| 🤐 Swallowing Difficulty | Trouble swallowing food or liquids |
| 🍷 Alcohol Consuming | Regular alcohol consumption |
| 😮‍💨 Coughing | Persistent or chronic cough |
| 🖐️ Yellow Fingers | Yellowing or staining of fingers |
| 💔 Chest Pain | Chest discomfort or pain |

### Disclaimer
⚠️ This tool is for **informational purposes only** and is **not** a substitute for
professional medical advice, diagnosis, or treatment. Always consult a qualified
healthcare provider with any questions you may have regarding a medical condition.
    """)
