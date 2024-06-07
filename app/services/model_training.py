import os
from os import getenv

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


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


def train_model_with_svd(print_predictions=False, limit_run=True):
    texts, labels = load_data(getenv("TRAINED_DATA_FOLDER"), limit_run=limit_run)

    # Creating a pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000, ngram_range=(1, 2), stop_words="english"
                ),
            ),
            ("svd", TruncatedSVD(n_components=100)),  # Dimensionality reduction
            ("normalizer", Normalizer()),  # Normalize data for SVM
            ("svm", SVC(kernel="linear", class_weight="balanced", C=1.0)),
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predictions and Evaluation
    predictions = pipeline.predict(X_test)
    if print_predictions:
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))

def train_model_with_svc(print_predictions=False, limit_run=True):
    # Load your data
    texts, labels = load_data(getenv("TRAINED_DATA_FOLDER"), limit_run=limit_run)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Feature extraction with bigrams
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # SVM Model
    model = SVC(kernel="linear", class_weight="balanced", C=1.0)
    model.fit(X_train_tfidf, y_train)

    # # Predictions and Evaluation
    # predictions = model.predict(X_test_tfidf)
    #
    # Cross-validate and predict
    predictions = cross_val_predict(model, X_train_tfidf, y_train, cv=5)
    report = classification_report(y_train, predictions, zero_division=0)
    if print_predictions:
        print(report)
        print(confusion_matrix(y_train, predictions))

    return report


def train_model_with_smote(print_predictions=False, limit_run=True):
    texts, labels = load_data(getenv("TRAINED_DATA_FOLDER"), limit_run=limit_run)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Creating a pipeline with SMOTE
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        SMOTE(random_state=42),
        SVC(kernel="linear", class_weight="balanced", C=1.0),
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predictions and Evaluation
    predictions = pipeline.predict(X_test)
    if print_predictions:
        print(classification_report(y_test, predictions, zero_division=0))
        print(confusion_matrix(y_test, predictions))



def train_model_with_random_forest(print_predictions=False, limit_run=True):
    # Load your data
    texts, labels = load_data(getenv("TRAINED_DATA_FOLDER"), limit_run)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Feature extraction with extended n-grams and adjusted TF-IDF parameters
    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 3), min_df=3, max_df=0.85
    )

    # Use RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,  # increased from 100
        max_depth=20,  # added max depth
        random_state=42,
        class_weight="balanced",
    )
    # Create a pipeline
    clf = make_pipeline(vectorizer, model)

    # Train the model
    clf.fit(X_train, y_train)

    # Predictions and evaluation
    predictions = clf.predict(X_test)
    if print_predictions:
        print(classification_report(y_test, predictions, zero_division=0))
        print(confusion_matrix(y_test, predictions))

    plot_feature_importances_save_file(clf.steps[1][1], clf.steps[0][1])


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
