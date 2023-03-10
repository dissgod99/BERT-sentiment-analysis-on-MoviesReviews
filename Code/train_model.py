import numpy as np
import pandas as pd
from preprocess_dataset import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import torch
from Dataset_Loader import Dataset


#Training constants
MAX_LENGTH = 512
LR = 2e-5
CHECKPOINT = "bert-base-cased"
BATCH_SIZE = 4
EPOCHS = 3

tokenizer = BertTokenizer.from_pretrained(CHECKPOINT)

def tokenize_fct(dataframe:pd.DataFrame):
    lst_text = list(dataframe["text"].values)
    out_tokenizer = tokenizer(
        lst_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return out_tokenizer

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__=="__main__":

    df_train = pd.read_csv(TRAIN_DIR)
    df_dev = pd.read_csv(DEV_DIR)
    df_test = pd.read_csv(TEST_DIR)

    out1 = tokenize_fct(df_train)
    print(out1)

    model = BertForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    
    train_encoded = tokenize_fct(df_train)
    dev_encoded = tokenize_fct(df_dev)
    test_encoded = tokenize_fct(df_test)

    args = TrainingArguments(
        #"Fine-tuned BERTModel",
        output_dir="output_classification",
        evaluation_strategy = "steps",
        save_strategy = "steps",
        logging_steps = 100,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

y_train = list(df_train["label"].values)
y_dev = list(df_dev["label"].values)
y_test = list(df_test["label"].values)

training_dataset = Dataset(train_encoded, y_train)
dev_dataset = Dataset(dev_encoded, y_dev)
testing_dataset = Dataset(test_encoded, y_test)

trainer = Trainer(
     model=model,
     args=args,
     train_dataset=training_dataset,
     eval_dataset=dev_dataset,
     tokenizer=tokenizer,
     compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()







