import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.features.preprocess import preprocess_texts
import yaml
from spacy.lang.en.stop_words import STOP_WORDS

# ---------------------------
# Page Config + Custom CSS
# ---------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="📰")

css_files = [
    "assets/styles/base.css",
    "assets/styles/components.css",
    "assets/styles/layout.css"
]
for css_file in css_files:
    try:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Missing CSS file: {css_file}")

# ---------------------------
# Load config + model
# ---------------------------
try:
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Missing configuration file: params.yaml")
    st.stop()

try:
    model = joblib.load("models/tfidf_logreg.joblib")
except FileNotFoundError:
    st.error("Model not found. Please train it first.")
    st.stop()

label_map = {"Fake": "Fake", "Real": "Real", "Satire": "Satire"}
colors = {"Real": "#27AE60", "Fake": "#E74C3C", "Satire": "#F39C12"}

# ---------------------------
# Header + Sidebar
# ---------------------------
st.title("📰 Fake News Detector")
st.markdown("Classify news articles as **Fake**, **Real**, or **Satire** using NLP and ML.")

st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
model_choice = st.sidebar.selectbox("Choose model", ["TF-IDF + Logistic Regression", "BERT (coming soon)"])
if model_choice == "BERT (coming soon)":
    st.sidebar.warning("BERT model not yet implemented.")

# Apply dark mode class
if dark_mode:
    st.markdown('<script>document.body.classList.add("dark-mode");</script>', unsafe_allow_html=True)
else:
    st.markdown('<script>document.body.classList.remove("dark-mode");</script>', unsafe_allow_html=True)

# ---------------------------
# Model Explanation toggle
# ---------------------------
st.subheader("🔍 Model Explanation")
option = st.radio("Choose how to display feature importance:", ("Hide", "Word Cloud", "Bar Chart"))

# ---------------------------
# Text input
# ---------------------------
st.subheader("📝 Single Article Prediction")
title = st.text_input("Title")
body = st.text_area("Body")

if st.button("Predict"):
    if not title.strip() and not body.strip():
        st.error("Please enter at least a title or body.")
        st.stop()

    try:
        text = title + " " + body
        processed = preprocess_texts([text], cfg)
        pred = model.predict(processed)[0]
        probs = model.predict_proba(processed)[0]
        class_index = list(model.classes_).index(pred)
        prob = probs[class_index]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
            <div class='prediction-card' style='background-color: {colors[pred]}'>
                Prediction: {label_map[pred]}
            </div>
        """, unsafe_allow_html=True)

        st.write(f"Confidence: {prob:.2f}")
        st.markdown(f"""
            <div class='confidence-bar'>
                <div class='confidence-fill' style='background-color: {colors[pred]}; width: {prob * 100:.1f}%'>
                    {prob * 100:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        df_probs = pd.DataFrame({
            "Class": model.classes_,
            "Probability": probs * 100
        })
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(df_probs["Class"], df_probs["Probability"],
               color=[colors.get(cls, "#999999") for cls in df_probs["Class"]],
               width=0.5)
        ax.set_ylabel("Probability (%)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # Explanation section
    # ---------------------------
    if option != "Hide":
        st.subheader("📊 Feature Importance")
        try:
            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            feature_names = tfidf.get_feature_names_out()
            coefs = clf.coef_[class_index]

            if len(tfidf.vocabulary_) == 0:
                st.warning("TF-IDF vocabulary is empty.")

            elif option == "Word Cloud":
                weights = {
                    term: weight
                    for term, weight in zip(feature_names, coefs)
                    if term.lower() not in STOP_WORDS
                }
                if not weights:
                    st.info("No strong features available.")
                else:
                    wc = WordCloud(
                        width=800, height=400,
                        background_color="white",
                        colormap="viridis",
                        prefer_horizontal=0.9,
                        max_words=100,
                        contour_color="steelblue",
                        contour_width=1
                    ).generate_from_frequencies(weights)
                    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    plt.tight_layout(pad=0)
                    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
                    st.pyplot(fig, clear_figure=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            elif option == "Bar Chart":
                filtered = [
                    (weight, term)
                    for weight, term in zip(coefs, feature_names)
                    if term.lower() not in STOP_WORDS
                ]
                top_pos = sorted(filtered, reverse=True)[:10]
                top_neg = sorted(filtered)[:10]
                df = pd.DataFrame(top_pos + top_neg, columns=["weight", "term"])

                if df.empty or df["weight"].abs().sum() == 0:
                    st.info("No strong features available.")
                else:
                    st.markdown(f"**Top Features for {label_map[pred]} Prediction**")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    bar_colors = [colors[pred] if w > 0 else "#999999" for w in df["weight"]]
                    df.plot.barh(x="term", y="weight", ax=ax, color=bar_colors)
                    ax.set_xlabel("Weight", fontsize=9)
                    ax.grid(axis="x", linestyle="--", alpha=0.4)
                    ax.tick_params(axis='both', labelsize=8)
                    plt.tight_layout(pad=0.5)
                    st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
                    st.pyplot(fig, clear_figure=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Explanation error: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("Built with ❤️ using Streamlit and Machine Learning | © 2024 Fake News Detector")
st.markdown("*Disclaimer: This tool is for educational purposes only. Always verify information from multiple sources.*")
st.markdown("</div>", unsafe_allow_html=True)