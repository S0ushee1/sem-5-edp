import joblib

# Load model once
model = joblib.load("models/tfidf_logreg.joblib")

def predict_texts(texts):
    """
    Predicts labels for a list of preprocessed texts.
    
    Args:
        texts (List[str]): List of preprocessed text strings.
    
    Returns:
        List[int]: Predicted class labels (e.g., 0=Fake, 1=Real, 2=Satire)
    """
    return model.predict(texts)