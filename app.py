import streamlit as st
import joblib
import re
import html
import numpy as np
import shap
import pytesseract
from PIL import Image
from scipy.sparse import hstack, csr_matrix

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️", layout="wide")

model = joblib.load('model_xgb.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')
structured_cols = joblib.load('structured_cols.joblib')

with st.sidebar:
    st.header("About this project")
    st.write(
        "This tool uses a machine learning model (XGBoost) trained on 17,880 real "
        "and fraudulent job postings to detect scam listings."
    )
    st.write("**Model performance:**")
    st.write("- Precision: 82%")
    st.write("- Recall: 82%")
    st.write("- PR-AUC: 0.91")
    st.markdown("---")
    st.caption("Built by Aniket Jaiswal")

st.title("🕵️ Fake Job Posting Detector")
st.write("Paste a job posting below, or upload a screenshot instead.")

col_left, col_right = st.columns(2)
with col_left:
    title = st.text_input("Job Title")
    description = st.text_area("Job Description", height=150)
with col_right:
    requirements = st.text_area("Requirements (optional)", height=100)
    company_profile = st.text_area("Company Profile (optional)", height=100)

st.markdown("---")
st.write("**Or upload a screenshot of the job posting instead:**")
uploaded_image = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"])

ocr_text = ""
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded screenshot", width=350)
    extracted_text = pytesseract.image_to_string(image)
    ocr_text = st.text_area("Extracted text (edit if needed)", value=extracted_text, height=150, key="ocr_text")

st.markdown("---")
st.write("**Additional signals:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    has_logo = st.checkbox("Company logo present?")
with col2:
    has_questions = st.checkbox("Screening questions included?")
with col3:
    telecommuting = st.checkbox("Remote / telecommuting role?")
with col4:
    has_salary = st.checkbox("Salary range listed?")

st.markdown("---")

if st.button("🔍 Check Posting", use_container_width=True):
    combined_typed_text = ' '.join([title, company_profile, description, requirements]).strip()

    if ocr_text.strip():
        source_text = combined_typed_text + ' ' + ocr_text
    else:
        source_text = combined_typed_text

    if not source_text.strip():
        st.warning("Please enter a job description or upload a screenshot.")
    else:
        def clean_text(text):
            text = html.unescape(text)
            text = text.lower()
            text = re.sub(r'<.*?>', ' ', text)
            text = re.sub(r'http\S+|www\S+', ' ', text)
            text = re.sub(r'[^a-z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        cleaned = clean_text(source_text)

        struct_vals = np.array([[
            int(telecommuting),
            int(has_logo),
            int(has_questions),
            int(has_salary),
            int(bool(company_profile.strip())),
        ]])

        text_vec = tfidf.transform([cleaned])
        X_input = hstack([text_vec, csr_matrix(struct_vals)])

        proba = model.predict_proba(X_input)[0, 1]

        result_col1, result_col2 = st.columns([1, 2])
        with result_col1:
            st.metric("Fraud Probability", f"{proba*100:.1f}%")
        with result_col2:
            if proba >= 0.5:
                st.error("🚩 This looks potentially fraudulent.")
            else:
                st.success("✅ This looks like a real posting.")

        st.markdown("### Why this prediction?")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input.toarray())

        feature_names = tfidf.get_feature_names_out().tolist() + structured_cols
        contributions = list(zip(feature_names, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        red_flags = [name for name, val in contributions if val > 0][:5]
        reassuring = [name for name, val in contributions if val < 0][:3]

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("#### 🚩 Signals toward *fraudulent*")
            with st.container(border=True):
                for flag in red_flags:
                    present = flag in cleaned
                    status = "found in text" if present else "notably absent"
                    st.write(f"- *{flag}* ({status})")

        with exp_col2:
            st.markdown("#### ✅ Signals toward *legitimate*")
            with st.container(border=True):
                for sign in reassuring:
                    present = sign in cleaned
                    status = "found in text" if present else "notably absent"
                    st.write(f"- *{sign}* ({status})")