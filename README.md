# Fake News Detector (EDP Project)

A machine learning–based fake news detection system that classifies news articles into **Fake**, **Real**, or **Satire**. It uses NLP preprocessing with a TF-IDF vectorizer and a Logistic Regression classifier, plus a Streamlit web UI and a Flask API for programmatic access.

## Features
- Single-article prediction via Streamlit UI
- REST API for predictions
- Configurable preprocessing and model parameters
- Baseline TF-IDF + Logistic Regression model

## Project Structure (Key Paths)
- `fake-news-detector/` main project folder
- `fake-news-detector/src/` training, prediction, preprocessing, evaluation
- `fake-news-detector/ui/streamlit_app.py` Streamlit app
- `fake-news-detector/src/api/app.py` Flask API
- `fake-news-detector/data/raw/` datasets
- `fake-news-detector/models/` trained model

## Setup
1. Open a terminal in the repo root.
2. Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r fake-news-detector/requirements.txt
```

## Train the Baseline Model
If the model file is missing or you want to retrain:

```bash
python -m fake-news-detector.src.models.train_baseline
```

The model is saved under `fake-news-detector/models/`.

## Run the Streamlit App

```bash
streamlit run fake-news-detector/ui/streamlit_app.py
```

Then open the local URL shown in the terminal.

## Run the API

```bash
python fake-news-detector/src/api/app.py
```

Example request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Sample Title","body":"Sample content"}'
```

## Notes
- Configuration lives in `fake-news-detector/params.yaml` and `fake-news-detector/config/params.yaml`.
- The dataset is under `fake-news-detector/data/raw/`.
- This project is for educational use.