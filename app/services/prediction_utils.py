import pickle


def _load_model_and_vectorizer():
    with open("stacked_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def predict_job_category(resume_text):
    model, vectorizer = _load_model_and_vectorizer()
    resume_tfidf = vectorizer.transform([resume_text])
    prediction = model.predict(resume_tfidf)
    return prediction[
        0
    ]  # Return the first prediction if only one document was processed
