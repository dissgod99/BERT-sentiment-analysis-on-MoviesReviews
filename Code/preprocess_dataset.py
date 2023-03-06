import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')

DATASET_DIR = "Dataset/"
TRAIN_DIR = "Dataset/Train.csv"
DEV_DIR = "Dataset/Valid.csv"
TEST_DIR = "Dataset/Test.csv"



def remove_line_breaks(text: str) -> str:
    out = re.sub(r'<br /><br />', '', text)
    return out

def remove_spoiler_alerts(text: str) -> str:
    tmp = re.sub(r'\*+.*\*+', '', text)
    tokenized_text = nltk.tokenize.word_tokenize(tmp)
    out = []
    for token in tokenized_text:
        if "spoiler" not in token.lower():
            out.append(token)
    return " ".join(out)

def load_df_as_pandas(directory:str, preprocess=False) -> pd.DataFrame:
    df = pd.read_csv(directory, sep=",")
    if preprocess:
        #dict_out = dict()
        sentiment = df["label"].apply(lambda x: "Positive" if x == 1 else "Negative")
        preprocessed_text = df["text"].apply(lambda x: remove_spoiler_alerts(remove_line_breaks(x)))
        #dict_out["text"] = preprocessed_text
        #dict_out["label"] = df["label"]
        #dict["sentiment"] = sentiment

        dict_out = {
            "text": preprocessed_text,
            "label": df["label"],
            "sentiment": sentiment
        }
        return pd.DataFrame(dict_out)
    return df




    


if __name__=="__main__":
    df = pd.read_csv(TRAIN_DIR)
    print(df["text"].iloc[104])
    print("++++++++++++++++++++++++++++++")
    print(remove_spoiler_alerts(remove_line_breaks(df["text"].iloc[104])))

    df_1 = load_df_as_pandas(TRAIN_DIR)
    print(df_1.head(20))
    