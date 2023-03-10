import numpy as np
from train_model import testing_dataset, y_test
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

MODEL_PATH = "output_classification/checkpoint-500"


if __name__=="__main__":
    model = BertForSequenceClassification(MODEL_PATH, num_labels=2)
    test_trainer = Trainer(model=model)
    raw_pred, _, _ = test_trainer.predict(testing_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    print(f"Predictions == {y_pred}")
    print(f"REPORT == {classification_report(y_test, y_pred)}")


