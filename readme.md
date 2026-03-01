# Lung Cancer Prediction

A Streamlit dashboard that uses a **Decision Tree Classifier** to estimate lung cancer risk based on six key symptoms. Toggle symptoms in the sidebar and the prediction updates instantly — no button required.

## Data Information
The model was trained on a dataset containing the following features:
- **Allergy**: History of allergies (Yes/No)
- **Swallowing Difficulty**: Difficulty in swallowing (Yes/No)
- **Alcohol Consuming**: Regular alcohol consumption (Yes/No)
- **Coughing**: Chronic coughing symptoms (Yes/No)
- **Yellow Fingers**: Presence of yellow stains on fingers (Yes/No)
- **Chest Pain**: Recurring chest pain (Yes/No)

---

## Running Locally

### Prerequisites
- Python 3.10 (matches the version pinned in `runtime.txt` for Streamlit Cloud)
- `pip` (comes with Python)

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/Haris-Mahboob-708/Lung_cancer_prediction.git
cd Lung_cancer_prediction

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501** in your browser.

---

## Deploying on Streamlit Community Cloud

Streamlit Community Cloud lets you host the app for free directly from your GitHub repository.

### Steps
1. **Push to GitHub** — make sure all files (`app.py`, `requirements.txt`, `runtime.txt`, `.streamlit/config.toml`, `decision_tree_model.joblib`, `lung_image.png`) are committed and pushed to your GitHub repository.

2. **Sign in to Streamlit Cloud** — go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

3. **Create a new app** — click **"New app"** (top-right corner).

4. **Fill in the deployment form**:
   | Field | Value |
   |---|---|
   | Repository | `Haris-Mahboob-708/Lung_cancer_prediction` |
   | Branch | `main` |
   | Main file path | `app.py` |

5. **Click "Deploy!"** — Streamlit Cloud will install dependencies from `requirements.txt` and start the app. Deployment usually takes 1–2 minutes.

6. **Share the URL** — once deployed, you'll get a public URL like `https://your-app-name.streamlit.app` that anyone can visit.

> **Tip:** The `runtime.txt` file in this repo pins Python 3.10 so Streamlit Cloud uses the correct interpreter automatically.

---

## Repository Structure
```
Lung_cancer_prediction/
├── app.py                        # Main Streamlit application
├── decision_tree_model.joblib    # Trained Decision Tree model
├── data.csv                      # Raw training dataset
├── lung_image.png                # Sidebar image
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for Streamlit Cloud
└── .streamlit/
    └── config.toml               # Streamlit server configuration
```

## Disclaimer
⚠️ This tool is for **informational purposes only** and is **not** a substitute for professional medical advice, diagnosis, or treatment.
