import os
from os import getenv
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import spacy
from app.utilities import clean_text
from app.services.resume_parser import preprocess_text
from app.utilities import extract_entities
from app.dataset import JOB_SKILLS


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    texts, labels = [], []
    for index, row in df.iterrows():
        category = row["Category"]
        resume_text = row["Resume"]
        cleaned_text = clean_text(resume_text)
        preprocessed_text = preprocess_text(cleaned_text)
        relevant_skills = " ".join(JOB_SKILLS.get(category, []))
        combined_text = preprocessed_text + " " + relevant_skills
        texts.append(combined_text)
        labels.append(category)
        print(f"loading... {category}:{combined_text[:70]}...")
    return texts, labels


def train_stacked_classifier(print_predictions=False):
    print(f"begin training with train_stacked_classifier. loading...")

    csv_path = os.getenv("RESUME_DATASET_PATH", "app.dataset/dataset.csv")
    texts, labels = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print(f"fitting vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.9
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"using SMOTE to balance data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

    base_learners = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=42, class_weight="balanced"
            ),
        ),
        ("svc", SVC(kernel="linear", probability=True)),
        ("dt", DecisionTreeClassifier(max_depth=20, random_state=42)),
    ]
    meta_learner = LogisticRegression(solver="lbfgs", random_state=42)
    stacked_model = StackingClassifier(
        estimators=base_learners, final_estimator=meta_learner, cv=5
    )
    print(f"fitting stacked model...")
    stacked_model.fit(X_train_resampled, y_train_resampled)

    with open(
        os.getenv("TRAINED_MODELS_PATH", "app/trained_models/") + "stacked_model.pkl",
        "wb",
    ) as model_file:
        pickle.dump(stacked_model, model_file)
    with open(
        os.getenv("TRAINED_MODELS_PATH", "app/trained_models/") + "vectorizer.pkl", "wb"
    ) as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    if print_predictions:
        print("printing predictions")
        predictions = stacked_model.predict(X_test_tfidf)
        print(classification_report(y_test, predictions, zero_division=0))
        print(confusion_matrix(y_test, predictions))
