import streamlit as st
import joblib
import numpy as np
import re
import shap
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_xgb.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    structured_features = joblib.load("structured_cols.joblib")
    return model, tfidf, structured_features

model, tfidf, structured_features = load_artifacts()

URGENCY_WORDS = ['immediate', 'urgent', 'no experience', 'earn from home',
                  'quick money', 'guaranteed', 'wire transfer', 'processing fee',
                  'work from home', 'easy money', 'be your own boss']


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def urgency_score(text):
    return sum(word in text for word in URGENCY_WORDS)


# ---------- UI ----------
st.title("🕵️ Fake Job Posting Detector")
st.caption("Paste a job posting below to check how likely it is to be fraudulent, with an explanation of why.")

with st.form("job_form"):
    title = st.text_input("Job Title", placeholder="e.g. Remote Data Entry Specialist")
    description = st.text_area("Job Description", height=150, placeholder="Paste the full job description here...")
    requirements = st.text_area("Requirements (optional)", height=80)
    company_profile = st.text_area("Company Profile (optional)", height=80)

    col1, col2, col3 = st.columns(3)
    with col1:
        has_logo = st.checkbox("Company logo present?", value=True)
    with col2:
        has_questions = st.checkbox("Screening questions included?")
    with col3:
        telecommuting = st.checkbox("Remote / telecommuting role?")

    salary_listed = st.checkbox("Salary range listed?")

    submitted = st.form_submit_button("Check Posting")

if submitted:
    if not description.strip():
        st.warning("Please paste a job description to analyze.")
    else:
        full_text = " ".join([title, company_profile, description, requirements])
        cleaned = clean_text(full_text)

        struct_vals = np.array([[
            int(telecommuting),
            int(has_logo),
            int(has_questions),
            int(salary_listed),
            int(bool(company_profile.strip())),
            0,  # has_benefits (not collected in simple form)
            int(bool(requirements.strip())),
            len(description),
            len(requirements),
            urgency_score(cleaned),
        ]])

        text_vec = tfidf.transform([cleaned])
        X_input = hstack([text_vec, csr_matrix(struct_vals)])

        proba = model.predict_proba(X_input)[0, 1]
        pred = "🚩 Likely Fraudulent" if proba >= 0.5 else "✅ Likely Real"

        st.subheader(pred)
        st.metric("Fraud Probability", f"{proba*100:.1f}%")
        st.progress(min(int(proba * 100), 100))

        # ---------- Explainability ----------
        st.markdown("### Why this prediction?")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input.toarray())

        feature_names = tfidf.get_feature_names_out().tolist() + structured_features
        contributions = list(zip(feature_names, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Friendly names for the structured (non-text) signals
        friendly_names = {
            "telecommuting": "this being a remote/work-from-home role",
            "has_company_logo": "a company logo",
            "has_questions": "screening questions for applicants",
            "has_salary": "a salary range listed",
            "has_company_profile": "a company profile/description",
        }

        def describe(feature, value, present):
            """Turn a raw feature + SHAP value into a plain-English sentence."""
            direction = "fraudulent" if value > 0 else "legitimate"

            if feature in friendly_names:
                label = friendly_names[feature]
                if present:
                    return f"This posting **has {label}**, which leans toward looking **{direction}**."
                else:
                    return f"This posting is **missing {label}**, which leans toward looking **{direction}**."
            else:
                if present:
                    return f"The posting uses the word/phrase **\"{feature}\"**, which leans toward looking **{direction}**."
                else:
                    return f"The posting does **not** use the word/phrase **\"{feature}\"**, which leans toward looking **{direction}**."

        red_flags = [(name, val) for name, val in contributions if val > 0][:5]
        reassuring = [(name, val) for name, val in contributions if val < 0][:3]

        if red_flags:
            st.markdown("**🚩 What made this look suspicious:**")
            for name, val in red_flags:
                if name in friendly_names:
                    idx = structured_features.index(name)
                    present = bool(struct_vals[0][idx])
                else:
                    present = name in cleaned
                st.write("- " + describe(name, val, present))

        if reassuring:
            st.markdown("**✅ What made this look legitimate:**")
            for name, val in reassuring:
                if name in friendly_names:
                    idx = structured_features.index(name)
                    present = bool(struct_vals[0][idx])
                else:
                    present = name in cleaned
                st.write("- " + describe(name, val, present))

        st.caption(
            "This explanation shows the words and details that most influenced the model's "
            "decision — both ones that were present in the posting, and ones that were "
            "notably missing."
        )

st.divider()
st.caption(
    "Built as a portfolio project. Model: XGBoost trained on the Kaggle "
    "'Real or Fake Job Posting Prediction' dataset. Not a substitute for your own judgment — "
    "always verify suspicious postings independently."
)