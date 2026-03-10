# src/eval/evaluate_baseline.py
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.data.loader import load_data
from src.features.preprocess import preprocess_texts

def explanatory_top_features(pipe, text, top_k=10):
    # Only for word vectorizer
    vect = pipe.named_steps["vect"].transform([text])
    clf = pipe.named_steps["clf"]
    classes = clf.classes_
    coefs = clf.coef_  # shape: [n_classes, n_features]
    # Feature names from FeatureUnion: get from each vectorizer
    word_feat_names = pipe.named_steps["vect"].transformer_list[0][1].get_feature_names_out()
    char_feat_names = pipe.named_steps["vect"].transformer_list[1][1].get_feature_names_out()
    feat_names = np.concatenate([word_feat_names, char_feat_names])

    # For predicted class, pick highest weights where feature present
    proba = None
    try:
        proba = clf.predict_proba(vect)[0]
    except Exception:
        proba = None
    pred_idx = clf.predict(vect)[0]
    pred_class_idx = np.where(classes == pred_idx)[0][0]
    weights = coefs[pred_class_idx]

    # Non-zero indices for this text
    nz = vect.nonzero()[1]
    scored = [(feat_names[i], weights[i]) for i in nz]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k], pred_idx, proba.max() if proba is not None else None

def main():
    train, val, test, cfg = load_data()
    X_test = preprocess_texts(test["text"].tolist(), cfg)
    y_test = test[cfg["data"]["label_field"]].tolist()

    pipe = joblib.load("models/tfidf_logreg.joblib")
    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred, digits=3))
    print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))

    cm = confusion_matrix(y_test, y_pred, labels=pipe.named_steps["clf"].classes_)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=pipe.named_steps["clf"].classes_,
                yticklabels=pipe.named_steps["clf"].classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Example explanation on one sample
    sample_text = X_test[0]
    top_feats, pred, conf = explanatory_top_features(pipe, sample_text)
    print("Pred:", pred, "Conf:", conf)
    print("Top contributing n-grams:", top_feats)

if __name__ == "__main__":
    main()