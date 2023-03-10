import numpy as np
import pandas as pd
from preprocess_dataset import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, trainer, TrainingArguments

MAX_LENGTH = 512

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def tokenize_fct(dataframe:pd.DataFrame):
    lst_text = list(dataframe["text"].values)
    return tokenizer(
        lst_text,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )



if __name__=="__main__":
    print("IT WORKS!!!")
    print("Added Conda + Libraries, e.g: transformers, sentence-transformers, sklearn, streamlit ...")
    df_train = pd.read_csv("Dataset/Train_preprocessed.csv")
    lst_text = list(df_train["text"].values)
    print(lst_text[:5])
    out1 = tokenize_fct(df_train)
    print(out1)