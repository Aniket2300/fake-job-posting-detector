# 🕵️ Fake Job Posting Detector

A machine learning system that detects fraudulent job postings, built to help job
seekers avoid scams and wasted time — inspired by the challenges of navigating job
listings during my own job search.

## Problem

Fraudulent postings are rare — only 4.84% of listings in this dataset are fake — which
makes this a classic imbalanced classification problem. A naive model that always predicts
"real" would score ~95% accuracy while catching zero fraud, so accuracy alone is a
misleading metric here. The project instead focuses on precision, recall, and PR-AUC.

## Approach

- Data: Kaggle's "Real or Fake Job Posting Prediction" dataset (17,880 postings)
- Text features: TF-IDF on job title, description, requirements, company profile
- Structured features: company logo presence, screening questions, salary listed, etc.
- Handled class imbalance using `class_weight='balanced'` (Logistic Regression) and
  `scale_pos_weight` (XGBoost)
- Compared two models: Logistic Regression vs XGBoost
- Explainability: SHAP values to show why a posting was flagged, shown directly in the app

## Bonus Feature: Screenshot Upload

Since many job postings are shared as screenshots (WhatsApp, LinkedIn app, etc.) rather
than plain text, the app supports uploading a screenshot directly. It uses Tesseract OCR
to extract the text automatically and pre-fills the description field, which the user can
review and correct before submitting (OCR isn't always perfect). From there it flows
through the same detection pipeline as manually typed input — no separate model or logic
needed.

## Results

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.51 | 0.91 | 0.858 |
| XGBoost | 0.82 | 0.80 | 0.913 |

XGBoost was chosen as the final model. While Logistic Regression catches slightly more
fraud (91% recall), it does so at the cost of wrongly flagging half of its "fraud" guesses
as false positives (51% precision) — a poor trade-off for real users. XGBoost strikes a
much better balance, correctly flagging fraud 82% of the time it raises an alert, while
still catching 80% of all actual fraud.

## Key Insight

The strongest single predictor wasn't a text feature — it was the absence of a company
logo. Real postings include a company logo 82% of the time, versus only 33% for fraudulent
postings, making it a stronger standalone signal than most text-based features.

## Known Limitation

SHAP analysis revealed the model partly relies on dataset-specific quirks (e.g., certain
names or wording patterns that happened to correlate with real postings in this particular
dataset) rather than purely generalizable fraud signals. This is a known risk when training
on a single dataset, and the vocabulary was manually adjusted to exclude the clearest cases
found during testing. A production system would need broader, more diverse training data
to fully address this.

In manual testing, this showed up as a false negative on a reshipping/banking-details style
scam (asks for banking info and a "processing fee" up front): the model scored it at 40.2%
fraud probability, below the 50% threshold, despite it matching several classic scam
patterns. TF-IDF scores words independently and doesn't capture that certain *combinations*
(e.g. asking for banking details + a fee together) are a stronger signal than either word
alone — likely because this dataset underrepresents this particular scam style relative to
more generic "urgent hiring, vague duties" scams. Addressing this would need either more
training examples of this scam type or an engineered feature that explicitly flags requests
for financial/banking information.

## Project Structure

```
Fake_Job_Posting/
├── fake_job_detector.ipynb   # Full analysis: EDA, feature engineering, modeling, SHAP
├── app.py                    # Streamlit app with text + screenshot upload
├── requirements.txt
├── model_xgb.joblib
├── tfidf_vectorizer.joblib
├── structured_cols.joblib
└── Data/
    └── fake_job_postings.csv
```

## Running Locally

```bash
pip install -r requirements.txt
jupyter notebook fake_job_detector.ipynb   # to explore the analysis
streamlit run app.py                        # to run the app
```

Requires [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed
separately for the screenshot upload feature.

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · SHAP · TF-IDF · Streamlit · Tesseract OCR (pytesseract)

---
*Built as part of my AI/ML portfolio during my MCA at Amity University Kolkata.*