import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.data.loader import load_data
from src.features.preprocess import preprocess_texts


def build_pipeline(cfg):
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(cfg["tfidf"]["word_ngram"]),
        max_features=cfg["tfidf"]["max_features"],
        min_df=1,          # ✅ ensure at least 1 occurrence is enough
        max_df=0.95,       # ✅ drop only words in >95% of docs
        sublinear_tf=cfg["tfidf"]["sublinear_tf"],
        stop_words=cfg["tfidf"].get("stop_words", "english")
    )
    clf = LogisticRegression(
        max_iter=cfg["logreg"]["max_iter"],
        C=cfg["logreg"]["C"],
        class_weight=cfg["logreg"]["class_weight"],
        n_jobs=-1
    )
    # ✅ Name the step "tfidf" so Streamlit can access it
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def main():
    train, val, test, cfg = load_data()
    print("Training set class distribution:")
    print(train[cfg["data"]["label_field"]].value_counts())
    X_train = preprocess_texts(train["text"].tolist(), cfg)
    X_val = preprocess_texts(val["text"].tolist(), cfg)
    y_train = train[cfg["data"]["label_field"]].tolist()
    y_val = val[cfg["data"]["label_field"]].tolist()

    print("🔧 Building and training pipeline...")
    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)

    # ✅ Safety check: ensure vocabulary is not empty
    vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)
    if vocab_size == 0:
        raise ValueError("❌ TF-IDF vocabulary is empty. Check params.yaml (min_df, stop_words).")
    print(f"✅ TF-IDF vocabulary size: {vocab_size}")

    preds = pipe.predict(X_val)
    print("📊 Validation Results:")
    print(classification_report(y_val, preds, digits=3))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/tfidf_logreg.joblib")
    print("💾 Model saved to models/tfidf_logreg.joblib")

    # 🔎 Debug: show top features for each class
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    for idx, class_label in enumerate(clf.classes_):
        coefs = clf.coef_[idx]
        top_pos = sorted(zip(coefs, feature_names), reverse=True)[:10]
        top_neg = sorted(zip(coefs, feature_names))[:10]

        print(f"\n📊 Top features for class '{class_label}':")
        print("  Positive weights (push toward this class):")
        for w, t in top_pos:
            print(f"    {t:20s} {w:.3f}")
        print("  Negative weights (push away from this class):")
        for w, t in top_neg:
            print(f"    {t:20s} {w:.3f}")


if __name__ == "__main__":
    main()