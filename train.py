import re
from datetime import datetime
from random import choice

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import ClassLabel, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate                        # make sure to install this via pip

MODEL_NAME = "distilbert-base-uncased" # microsoft/deberta-v3-base
FINAL_MODEL_SUBDIR = "final"
BATCH_SIZE = 32
MAX_LEN = 128
LR = 5e-5                              # 2e-5 is usually better, possibly slower training
EPOCHS = 3                             # 5
SEED = 42                              # reproducibility, e.g. can be 67
TEST_SIZE = 0.05
CHECKPOINT_PATH = "./checkpoints"

ds = load_dataset("Jacobvs/PoliticalTweets")

latest = max(ds["train"]["date"],
             key=lambda s: datetime.fromisoformat(s)) # 2023-02-19 23:32:00
print(f"The most recent tweet on the dataset was made on: {latest}")

pattern = re.compile(r" https://t\.co/\S+") # link pattern

def clean_example(example):
    text = example["text"]
    text = pattern.sub("", text)
    text = text.replace("&amp;", "&")
    example["text"] = text
    return example

ds["train"] = ds["train"].map(clean_example) # num_proc=4
ds = ds["train"].train_test_split(test_size=TEST_SIZE, seed=SEED)

device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(
    torch.backends, 'mps'
) and torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

labels = sorted(set(ds["train"]["party"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

texts = ds["train"]["text"]     # for train, this is just a check, can use this in a better do_tokenize
token_counts = [len(tokenizer.tokenize(t)) for t in texts]

mean_tokens = np.mean(token_counts)
max_tokens = max(token_counts)
print(f"mean, max tokens: {mean_tokens:.2f}, {max_tokens}")

# basic ver
def do_tokenize(batch):
    encoding = tokenizer(batch["text"], max_length=MAX_LEN,
                         padding="max_length", truncation=True)
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "label": label2id[batch["party"]]
    }

tokenized = ds.map(do_tokenize, remove_columns=[
    c for c in ds["train"].column_names if c not in ("label", "text")
])

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy.compute(predictions=preds, references=p.label_ids)
    f1m = f1.compute(predictions=preds, references=p.label_ids, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1m["f1"]}

training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=1,
    report_to="none",
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("validation", tokenized["test"]),
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

final_path = f"{CHECKPOINT_PATH}/{FINAL_MODEL_SUBDIR}"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

# make sure it is saved properly and can thus be loaded
tokenizer = AutoTokenizer.from_pretrained(final_path)

model = AutoModelForSequenceClassification.from_pretrained(
    final_path,
    num_labels=len(label2id), # not really needed for after final_path, keep for good measure
    id2label=id2label,
    label2id=label2id,
).to(device)

def predict(texts):
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt").to(device)
    outputs = model(**enc)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(preds)), preds]
    return [{"label": id2label[p], "confidence": float(conf)} for p, conf in zip(preds, confidences)]

predict("Gas prices")             # -> [{'label': 'Republican', 'confidence': 0.9958134293556213}]
predict("Medicaid")               # -> [{'label': 'Democrat', 'confidence': 0.7480603456497192}]
