import streamlit as st
import os
import pandas as pd
import pickle
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from app.services import process_resume, preprocess_text
from app.utilities import extract_top_skills, clean_text
from app.dataset import JOB_SKILLS

# Set up the page title and layout
st.set_page_config(page_title="Resume Parser", layout="wide")


# Load the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open("app/trained_models/stacked_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("app/trained_models/vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


# Predict job category
def predict_job_category(resume_text):
    model, vectorizer = load_model_and_vectorizer()
    resume_tfidf = vectorizer.transform([resume_text])
    prediction = model.predict(resume_tfidf)
    return prediction[
        0
    ]  # Return the first prediction if only one document was processed


# Training the model
def train_stacked_classifier():
    st.write("Training the model...")
    csv_path = "app/dataset/dataset.csv"
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

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.9
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
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
    stacked_model.fit(X_train_resampled, y_train_resampled)

    with open("app/trained_models/stacked_model.pkl", "wb") as model_file:
        pickle.dump(stacked_model, model_file)
    with open("app/trained_models/vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    predictions = stacked_model.predict(X_test_tfidf)
    report = classification_report(y_test, predictions, zero_division=0)
    st.write(report)
    st.write(confusion_matrix(y_test, predictions))


# Streamlit interface
st.title("Resume Parser")

# Upload Resume Section
st.header("Upload New File")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
if uploaded_file is not None:
    file_details = {
        "filename": uploaded_file.name,
        "filetype": uploaded_file.type,
        "filesize": uploaded_file.size,
    }
    st.write(file_details)
    # Ensure the directory exists
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")
    # Save the file to a temporary location
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Process the resume
    filepath = os.path.join("tempDir", uploaded_file.name)
    preprocessed_text = process_resume(filepath)["preprocessed_text"]
    prediction = predict_job_category(preprocessed_text)
    top_skills = extract_top_skills(preprocessed_text, prediction)
    st.write(f"Prediction: {prediction}")
    st.write(f"Top Skills: {top_skills}")

# Train Model Section
st.header("Train Model")
if st.button("Train Model"):
    train_stacked_classifier()
    st.success("Model trained successfully!")
