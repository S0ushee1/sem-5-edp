# src/features/preprocess.py
import re
import spacy
from typing import List
from spacy.lang.en.stop_words import STOP_WORDS
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')

# Load small spaCy model; disable parser for speed
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    return _nlp

def clean_text(text: str, lowercase=True, remove_urls=True, remove_emails=True):
    if remove_urls:
        text = URL_RE.sub(" ", text)
    if remove_emails:
        text = EMAIL_RE.sub(" ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    if lowercase:
        text = text.lower()
    return text

def lemmatize(text: str):
    nlp = get_nlp()
    doc = nlp(text)
    return " ".join(
        t.lemma_ for t in doc
        if not t.is_space and not t.is_punct and t.lemma_.lower() not in STOP_WORDS
    )

def preprocess_texts(texts: List[str], cfg):
    out = []
    for t in texts:
        s = clean_text(t, cfg["preprocess"]["lowercase"],
                       cfg["preprocess"]["remove_urls"],
                       cfg["preprocess"]["remove_emails"])
        if cfg["preprocess"]["lemmatize"]:
            s = lemmatize(s)
        out.append(s)
    return out