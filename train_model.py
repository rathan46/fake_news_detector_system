import os
import pickle
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from text_utils import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "fake_or_real_news.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found. Expected file at: data/fake_or_real_news.csv"
        )

    dataframe = pd.read_csv(DATASET_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"].str.len() > 20].copy()
    dataframe["clean_text"] = dataframe["text"].apply(clean_text)
    dataframe = dataframe[dataframe["clean_text"].str.len() > 0].copy()
    dataframe["label_value"] = dataframe["label"].str.upper().map({"FAKE": 0, "REAL": 1})
    dataframe = dataframe.dropna(subset=["label_value"])
    return dataframe


def train_and_save():
    dataframe = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["clean_text"],
        dataframe["label_value"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["label_value"],
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train_vectorized, y_train)

    predictions = model.predict(x_test_vectorized)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["FAKE", "REAL"])

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model training completed successfully.")
    print(f"Dataset rows used: {len(dataframe)}")
    print(f"Class distribution: {dict(Counter(dataframe['label'].str.upper()))}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification report:")
    print(report)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_and_save()
