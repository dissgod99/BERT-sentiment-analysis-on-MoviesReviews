import numpy as np
import pandas as pd
from preprocess_dataset import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, trainer, TrainingArguments

MAX_LENGTH = 512

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


if __name__=="__main__":
    df_train = pd.read_csv("Dataset/Train.csv")
    print(df_train)
    print(list(df_train["text"].values[:10]))
    first = list(df_train["text"].values)
    out_tokenizer = tokenizer(
        first,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    print(out_tokenizer)