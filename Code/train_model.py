import numpy as np
import pandas as pd
from preprocess_dataset import *


if __name__=="__main__":
    print("IT WORKS!!!")
    print("Added Conda + Libraries, e.g: transformers, sentence-transformers, sklearn, streamlit ...")
    a = np.array([1, 2, 3])
    print(a.sum())
    """df = load_df_as_pandas(TRAIN_DIR)
    print(df.head(20))"""

    df_1 = load_df_as_pandas(TRAIN_DIR, preprocess=True)
    print(df_1.head(110))