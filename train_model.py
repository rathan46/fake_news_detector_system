import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from comparison_utils import extract_comparison_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATASET_PATH = os.path.join(BASE_DIR, "data", "fake_or_real_news.csv")
DATASET_PATH = os.path.join(BASE_DIR, "data", "news_api_comparison_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def load_source_dataset():
    if not os.path.exists(SOURCE_DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found. Expected file at: data/fake_or_real_news.csv"
        )

    dataframe = pd.read_csv(SOURCE_DATASET_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "Source dataset must contain 'text' and 'label' columns. "
            "Use data/fake_or_real_news.csv as the raw source corpus."
        )

    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"].str.len() > 20].copy()
    return dataframe


def build_supportive_variants(text):
    words = str(text).split()
    if not words:
        return []

    lead_length = min(max(8, len(words) // 2), len(words))
    lead = " ".join(words[:lead_length]).strip()
    full_text = " ".join(words).strip()
    summary = f"{lead}.".strip()

    variants = []
    for item in [full_text, summary, f"{lead}. {full_text}".strip()]:
        if item and item not in variants:
            variants.append(item)
    return variants


def build_comparison_dataset():
    source_dataframe = load_source_dataset().reset_index(drop=True)
    rows = []

    for index, row in source_dataframe.iterrows():
        claim_text = row["text"]

        for evidence_text in build_supportive_variants(claim_text):
            rows.append(
                {
                    "claim_text": claim_text,
                    "evidence_text": evidence_text,
                    "support_label": 1,
                    "pair_type": "supportive",
                }
            )

        for offset in range(1, 4):
            mismatch_index = (index + offset) % len(source_dataframe)
            mismatch_text = source_dataframe.iloc[mismatch_index]["text"]
            mismatch_variants = build_supportive_variants(mismatch_text)
            for evidence_text in mismatch_variants[:2]:
                rows.append(
                    {
                        "claim_text": claim_text,
                        "evidence_text": evidence_text,
                        "support_label": 0,
                        "pair_type": "mismatch",
                    }
                )

    comparison_dataframe = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    comparison_dataframe.to_csv(DATASET_PATH, index=False)
    return comparison_dataframe


def train_and_save():
    comparison_dataframe = build_comparison_dataset()
    feature_matrix = np.array(
        [
            extract_comparison_features(row["claim_text"], row["evidence_text"])
            for _, row in comparison_dataframe.iterrows()
        ]
    )
    labels = comparison_dataframe["support_label"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["MISMATCH", "SUPPORT"])

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    print("Comparison model training completed successfully.")
    print(f"Training dataset saved to: {DATASET_PATH}")
    print(f"Dataset rows used: {len(comparison_dataframe)}")
    print(f"Pair distribution: {dict(Counter(comparison_dataframe['pair_type']))}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification report:")
    print(report)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
