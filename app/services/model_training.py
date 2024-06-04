import os
from os import getenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC


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


def train_model(print_predictions=False, limit_run=True):
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

    # return model, vectorizer
    return report
