import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import re

# -------------------------------
# Load trained model
model = load_model('resume_classifier_model.h5')

# Load TF-IDF and LabelEncoder
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("LabelEncoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------------
# Text cleaning function
def clean_input(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# Custom CSS Styling
st.markdown("""
    <style>
    
   @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(to bottom right, #ffe6ec, #ffd9e8);
        padding: 2rem;
    }


    h1.title {
    font-family: 'Raleway', sans-serif !important;
    font-weight: 700;
    color: #e91e63;
    font-size: 3rem;
    text-align: center;
    margin-bottom: 1rem;
    line-height: 1.1;
    letter-spacing: 0.02em; /* thoda kam ker diya hai */
    word-spacing: 0.1em;    /* agar zyada space ho to kam ker sakte hain */
    text-shadow: 1px 1px 3px rgba(233, 30, 99, 0.3);
    }



    .stTextArea textarea {
        background-color: #fff !important;
        border: 2px solid #e91e63 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        font-size: 16px !important;
        color: #2c3e50 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        transition: box-shadow 0.3s ease-in-out, border-color 0.3s ease-in-out;
    }

    .stTextArea textarea:focus {
        border-color: #ff4f81 !important;
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.2) !important;
        outline: none !important;
    }

    .stButton > button {
        background: linear-gradient(to right, #e91e63, #ff6f91);
        border: none;
        border-radius: 10px;
        color: white;
        font-size: 18px;
        padding: 12px 30px;
        margin-top: 1rem;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 14px rgba(233, 30, 99, 0.3);
        cursor: pointer;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #ff6f91, #e91e63);
        box-shadow: 0 6px 18px rgba(233, 30, 99, 0.4);
    }

    .stAlert {
        border-radius: 12px;
        font-size: 18px;
        font-weight: 500;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Streamlit App UI

# Main heading
st.markdown("""
    <h1 style="
        text-align: center;
        color: #e91e63;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        line-height: 1.1;
        letter-spacing: 0.05em;
        font-family: 'Montserrat', sans-serif;
        text-shadow: 1px 1px 3px rgba(233, 30, 99, 0.3);
    ">
        💼 AI-Powered Resume Screening
    </h1>
""", unsafe_allow_html=True)

# Description lines
st.markdown("""
    <p style="color:#e91e63; font-size:18px; text-align:center; margin-bottom:0.2rem;">
        📝 Paste your resume below to predict its job category using artificial intelligence.
    </p>
    <p style="color:#e91e63; font-size:18px; text-align:center; margin-top:0;">
        🚀 Our AI model analyzes your text to identify the most relevant job field.
    </p>
""", unsafe_allow_html=True)

# Custom styled label
st.markdown(
    "<p style='color:#2c3e50; font-size:18px; font-weight:600; margin-top:1.5rem;'>📄 Paste your full resume here:</p>",
    unsafe_allow_html=True
)

# Text area without label
resume_text = st.text_area(
    label="",  # No label here
    placeholder="Include skills, experience, education, projects, etc...",
    height=300
)


# Centered Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Predict Category"):
        if resume_text.strip() != "":
            resume_text_clean = clean_input(resume_text)
            X_new = tfidf.transform([resume_text_clean]).toarray()
            pred_class = model.predict(X_new).argmax()
            pred_category = le.inverse_transform([pred_class])
            st.markdown(f"""
                <div style="
                    background-color: #ffe6ec;
                    border: 2px solid #e91e63;
                    color: #e91e63;
                    padding: 15px 20px;
                    border-radius: 12px;
                    font-weight: 700;
                    font-size: 20px;
                    margin-top: 1rem;
                    text-align: center;
                ">
                    ✅ Predicted Category: <strong>{pred_category[0]}</strong>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter your full resume text to get a prediction.")
  