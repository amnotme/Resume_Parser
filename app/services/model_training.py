import os
from os import getenv

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import numpy as np


def load_data(directory, limit_run=True):
    texts, labels = [], []

    if limit_run:
        limit_run_value = int(getenv("LIMIT_RUN", 10))
        for filename in os.listdir(directory):
            if filename.endswith(".txt") and limit_run_value > 0:
                filepath = os.path.join(directory, filename)
                file_name_split = filename.split("-")
                file_name_split.pop()
                job_type = "-".join(file_name_split)
                with open(filepath, "r", encoding="utf-8") as file:
                    texts.append(file.read())
                    labels.append(job_type)
                limit_run_value -= 1
            else:
                break
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                file_name_split = filename.split("-")
                file_name_split.pop()
                job_type = "-".join(file_name_split)  # Extract job type from filename
                with open(filepath, "r", encoding="utf-8") as file:
                    texts.append(file.read())
                    labels.append(job_type)
    return texts, labels


def train_stacked_classifier(print_predictions=False, limit_run=True):

    print(f"begin training with: {__name__}")
    # Load your data
    texts, labels = load_data(getenv("TRAINED_DATA_FOLDER"), limit_run)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 3), min_df=3, max_df=0.85
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

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
    stacked_model.fit(X_train_tfidf, y_train)
    predictions = stacked_model.predict(X_test_tfidf)
    if print_predictions:
        print(classification_report(y_test, predictions, zero_division=0))
        print(confusion_matrix(y_test, predictions))

    # Assuming RandomForest is the first model in the base_learners
    rf_model = stacked_model.named_estimators_["rf"]
    plot_feature_importances_save_file(rf_model, vectorizer)


def train_stacked_classifier_with_word2vec(print_predictions=False, limit_run=True):
    # Load your data
    texts, labels = load_data(getenv("TRAINED_DATA_FOLDER"), limit_run)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Prepare tokenized data for Word2Vec
    X_train_tokenized = [word_tokenize(text.lower()) for text in X_train]
    X_test_tokenized = [word_tokenize(text.lower()) for text in X_test]

    # Create and train Word2Vec model
    vectorizer = Word2Vec(sentences=X_train_tokenized, vector_size=100, window=5, min_count=5, workers=4)

    # Transform text data to vector data
    X_train_vectors = [np.mean([vectorizer.wv[word] for word in words if word in vectorizer.wv.key_to_index], axis=0)
                       for words in X_train_tokenized]
    X_test_vectors = [np.mean([vectorizer.wv[word] for word in words if word in vectorizer.wv.key_to_index], axis=0) for
                      words in X_test_tokenized]

    # Define base learners
    base_learners = [
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)),
        ("svc", SVC(kernel="linear", probability=True)),
        ("dt", DecisionTreeClassifier(max_depth=10, random_state=42))
    ]

    # Meta-learner
    meta_learner = LogisticRegression(random_state=42)

    # Stacking Classifier
    stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

    # Train the model
    stacked_model.fit(np.array(X_train_vectors), y_train)

    # Predictions and evaluation
    predictions = stacked_model.predict(np.array(X_test_vectors))
    if print_predictions:
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))


def plot_feature_importances_save_file(
    model, vectorizer, n_features=20, file_name="feature_importances.png"
):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
        indices = np.argsort(importances)[-n_features:]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(10, 8))
        plt.title("Top Feature Importances", fontsize=16)
        bars = plt.barh(
            range(n_features), sorted_importances, align="center", color="skyblue"
        )
        plt.yticks(range(n_features), sorted_feature_names, fontsize=12)
        plt.xlabel("Relative Importance", fontsize=14)
        plt.ylabel("Features", fontsize=14)

        for bar in bars:
            plt.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center",
                ha="left",
                fontsize=10,
                color="blue",
            )

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()
    else:
        print("The model does not support feature importances.")
