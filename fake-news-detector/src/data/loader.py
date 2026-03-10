# src/data/loader.py
import pandas as pd
import yaml
import json
from pathlib import Path

def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_label_map():
    with open("assets/label_map.json", "r") as f:
        return json.load(f)

def read_split(path: str, label_map: dict, label_field: str):
    df = pd.read_csv(path)
    df[label_field] = df[label_field].map(lambda x: label_map.get(str(x), x))
    df = df.dropna(subset=[label_field])
    return df

def concat_text(row, fields):
    parts = [str(row.get(f, "")) for f in fields if pd.notna(row.get(f, ""))]
    return " . ".join([p.strip() for p in parts if p.strip()])

def load_data():
    cfg = load_config()
    label_map = load_label_map()
    label_field = cfg["data"]["label_field"]
    fields = cfg["data"]["text_fields"]

    train = read_split(cfg["data"]["train_path"], label_map, label_field)
    val = read_split(cfg["data"]["val_path"], label_map, label_field)
    test = read_split(cfg["data"]["test_path"], label_map, label_field)

    for df in (train, val, test):
        df["text"] = df.apply(lambda r: concat_text(r, fields), axis=1)

    return train, val, test, cfg