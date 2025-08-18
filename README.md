# 💼 AI-Powered Resume Screening System

An AI-based web app to automatically classify resumes into relevant job categories using **Natural Language Processing (NLP)** and a **Deep Learning model**. Built with Streamlit, Keras, and Scikit-learn.

---

## 🚀 Features

- 🔍 Predicts job category from raw resume text
- ✨ Clean and modern Streamlit UI with custom CSS
- 📊 WordCloud visualization of job categories
- 🧹 Resume text cleaning and preprocessing
- 🧠 Deep learning classification using Keras
- 💾 Model, TF-IDF, and LabelEncoder saved and reused

---

## 🗂 Dataset

**File:** `UpdatedResumeDataSet.csv`  
**Columns:**
- `Resume`: Raw resume text  
- `Category`: Job field/category (e.g., Data Science, HR, etc.)

---

## ⚙️ How It Works

1. **Text Cleaning:**
   - Lowercasing
   - Removing punctuation & special characters
   - Stopword removal
   - Lemmatization

2. **Feature Extraction:**
   - TF-IDF Vectorizer (`max_features=5000`)

3. **Model:**
   - Trained Keras deep learning model saved as `resume_classifier_model.h5`
   - Label Encoding used for category mapping

4. **Streamlit App:**
   - User pastes resume in text area
   - Resume is preprocessed and vectorized
   - Model predicts category
   - Result is shown with styled output

---

## 📁 Project Structure

## 📁 Project Structure

```
📦 ResumeScreeningAI/
├── app.py                     # Main Streamlit app
├── UpdatedResumeDataSet.csv   # Dataset file
├── tfidf.pkl                  # Saved TF-IDF vectorizer
├── LabelEncoder.pkl           # Saved LabelEncoder
├── resume_classifier_model.h5 # Trained Keras model
└── README.md                  # This file
```





---

## 📌 Future Improvements

Upload support for PDF/DOCX resumes

Highlight keywords based on category

Use advanced models like BERT

Deploy to Streamlit Cloud / HuggingFace Spaces


---
## 🧠 Sample Prediction Flow

Paste your resume text in the app

Resume is cleaned and vectorized

Deep learning model predicts job category

Output is displayed in a styled box


 
