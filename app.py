from pathlib import Path

import joblib
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment.config import ARTIFACTS_DIR, LABEL_NAMES
from sentiment.preprocessing import clean_review

st.set_page_config(page_title="Review Sentiment", page_icon="🛍️", layout="centered")
st.title("Customer Review Sentiment")
st.caption("Amazon product review polarity classifier")

MODEL_OPTIONS = {
    "Logistic Regression": ARTIFACTS_DIR / "models" / "tf-idf_logistic_regression.joblib",
    "Linear SVM": ARTIFACTS_DIR / "models" / "tf-idf_linear_svm.joblib",
    "DistilBERT": ARTIFACTS_DIR / "models" / "distilbert",
}


@st.cache_resource
def load_baseline(path: str):
    return joblib.load(path)


@st.cache_resource
def load_transformer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


model_choice = st.selectbox("Model", list(MODEL_OPTIONS))
review = st.text_area("Customer review", height=180, placeholder="Paste a product review...")

if st.button("Classify review", type="primary", use_container_width=True):
    cleaned = clean_review(review)
    if not cleaned:
        st.warning("Enter a review before classifying.")
    else:
        model_path = MODEL_OPTIONS[model_choice]
        if not model_path.exists():
            st.error("This model is not trained yet. Follow the training steps in README.md.")
        elif model_choice == "DistilBERT":
            tokenizer, model = load_transformer(str(model_path))
            inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=192)
            with torch.inference_mode():
                probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
            label_id = int(torch.argmax(probabilities))
            st.subheader(LABEL_NAMES[label_id].title())
            st.metric("Model score", f"{float(probabilities[label_id]):.1%}")
            st.caption("Softmax score; not calibrated probability.")
            st.bar_chart(
                {LABEL_NAMES[index]: float(probabilities[index]) for index in range(len(LABEL_NAMES))}
            )
        else:
            model = load_baseline(str(model_path))
            prediction = int(model.predict([cleaned])[0])
            st.subheader(LABEL_NAMES[prediction].title())
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([cleaned])[0]
                st.metric("Model score", f"{float(probabilities[prediction]):.1%}")
                st.bar_chart(
                    {
                        LABEL_NAMES[int(label)]: float(probability)
                        for label, probability in zip(model.classes_, probabilities)
                    }
                )
            else:
                st.info("Linear SVM does not provide calibrated confidence scores.")
