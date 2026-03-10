# src/models/train_bert.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_decay
from sklearn.metrics import classification_report
from src.data.loader import load_data

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = texts
        self.labels = [label2id[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def main():
    train, val, test, cfg = load_data()
    labels = cfg["data"]["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg["bert"]["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["bert"]["model_name"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    train_ds = NewsDataset(train["text"].tolist(), train["label"].tolist(), tokenizer, cfg["bert"]["max_len"], label2id)
    val_ds = NewsDataset(val["text"].tolist(), val["label"].tolist(), tokenizer, cfg["bert"]["max_len"], label2id)

    train_dl = DataLoader(train_ds, batch_size=cfg["bert"]["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["bert"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["bert"]["lr"])
    num_epochs = cfg["bert"]["epochs"]
    num_steps = num_epochs * len(train_dl)
    scheduler = get_linear_schedule_with_decay(optimizer, num_warmup_steps=int(0.1 * num_steps), num_training_steps=num_steps)

    best_val = -1
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                pred = out.logits.argmax(dim=1).cpu().numpy().tolist()
                true = batch["labels"].cpu().numpy().tolist()
                preds.extend(pred)
                trues.extend(true)
        # Macro-F1 via sklearn
        print(f"Epoch {epoch+1}:")
        print(classification_report(trues, preds, target_names=labels, digits=3))

    model.save_pretrained("models/distilbert_fake_news")
    tokenizer.save_pretrained("models/distilbert_fake_news")

if __name__ == "__main__":
    main()