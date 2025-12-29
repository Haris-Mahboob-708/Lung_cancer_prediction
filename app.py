import streamlit as st
import joblib
import numpy as np

# 1. Load the model
# We use the new filename you provided
try:
    model = joblib.load('decision_tree_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'decision_tree_model.joblib' is in the same folder.")
    st.stop()

# 2. App Title and Description
st.set_page_config(page_title="Health Prediction System", layout="wide")

st.title("Health Condition Predictor")
st.markdown("""
This application uses a Machine Learning model (Decision Tree) to predict health conditions based on symptoms and habits.
**Adjust the parameters in the sidebar** to see the prediction.
""")

# 3. Sidebar for Inputs
st.sidebar.header("Patient Symptoms & Habits")
st.sidebar.write("Select the status for each:")

# Helper function to create Yes/No inputs
def create_input(label):
    # Returns 0 for No, 1 for Yes (Adjust this if your model was trained on 1=No, 2=Yes)
    response = st.sidebar.radio(label, ["No", "Yes"], horizontal=True)
    return 1 if response == "Yes" else 0

# Create inputs for the 6 specific features found in your model
allergy = create_input("Allergy")
swallowing = create_input("Swallowing Difficulty")
alcohol = create_input("Alcohol Consuming")
coughing = create_input("Coughing")
fingers = create_input("Yellow Fingers")
chest_pain = create_input("Chest Pain")

# 4. Main Prediction Logic
st.divider()

if st.button("Predict Result"):
    # Create the input array in the exact order found in the model file
    input_data = np.array([[allergy, swallowing, alcohol, coughing, fingers, chest_pain]])
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result with a nice style like your screenshot
        st.subheader("Prediction Result:")
        
        # Logic to display friendly output (assuming prediction is a class label)
        if prediction == 1 or prediction == "YES" or str(prediction).upper() == "YES":
             st.error(f"Prediction: Positive (Condition Detected)")
        else:
             st.success(f"Prediction: Negative (Healthy/Normal)")
             
        # Show raw output just in case
        st.caption(f"Raw Model Output: {prediction}")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")