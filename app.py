import streamlit as st
import os
import pandas as pd
import pickle
import bcrypt
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from app.services import process_resume
from app.services import preprocess_text, predict_job_category
from app.utilities import extract_top_skills, clean_text
from app.dataset import JOB_SKILLS
import streamlit_authenticator as stauth
import logging

logging.basicConfig(level=logging.INFO)
# Set up the page title and layout
st.set_page_config(
    page_title="Resume Parser", layout="wide", initial_sidebar_state="expanded"
)

# User Authentication
names = ["John Doe", "Jane Doe", "Admin User"]
usernames = ["johndoe", "janedoe", "admin"]
passwords = ["password123", "password456", "adminpassword"]

hashed_passwords = [
    bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    for password in passwords
]

credentials = {
    "usernames": {
        usernames[i]: {"name": names[i], "password": hashed_passwords[i]}
        for i in range(len(usernames))
    }
}

authenticator = stauth.Authenticate(
    credentials, "resume_parser", "abcdef", cookie_expiry_days=30
)

# Login Form
fields = {
    "Form name": "Login",
    "Username": "Username",
    "Password": "Password",
    "Login": "Login",
}
name, authentication_status, username = authenticator.login("main", fields=fields)


# Sidebar content visible to all users
def set_sidebar() -> None:
    with st.sidebar:
        st.header("📄 About the Resume Parser")
        st.write(
            """
            Welcome to the Resume Parser app! This tool helps you analyze and categorize resumes based on their content.
            
            This parser focuses in Tech related resumes for the following job types:
            
            - Backend Developer
            - Cloud Engineer
            - Data Scientist
            - Frontend Developer
            - Full Stack Developer
            - Machine Learning Engineer
            - Mobile App Developer (iOS/Android)
            - Python Developer
             

            **Highlights**:

            - **Upload and Analyze**: Upload your resume in PDF format and get detailed insights.
            - **Interactive Visuals**: Explore word clouds, skill distributions, and more.
            - **Machine Learning**: Leverages advanced machine learning models for accurate predictions.

            **Login Instructions**:
            - Username: johndoe
            - Password: password123
            """
        )
        st.divider()

        st.header("👨‍💻 About the Author")
        st.write(
            """
            Leo is a software engineer based in NYC with a background in computer science from WGU. Working in tech, Leo is dedicated to contributing to innovative projects while pursuing the goal of starting his own business in the tech industry. Passionate about programming and technology trends, Leo enjoys engaging with the tech community to share insights and discuss code.

            Connect and say hi!
            """
        )

        st.divider()
        st.subheader("🔗 Connect with Me", anchor=False)
        st.markdown(
            """
            - [🐙 Source Code](https://github.com/amnotme/Resume_Parser)
            - [👔 LinkedIn](https://www.linkedin.com/in/leopoldo-hernandez/)
            """
        )

        st.divider()
        st.write("Made with ♥, powered by Streamlit")


if authentication_status:
    st.sidebar.write(f"Welcome {name}!")
    authenticator.logout("Logout", "sidebar")
elif authentication_status == False:
    st.sidebar.error("Username/password is incorrect")
elif authentication_status == None:
    st.sidebar.warning("Please enter your username and password")

if authentication_status:

    # Initialize session state for uploaded resume
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # Training the model
    def train_stacked_classifier():
        st.write("Training the model...")
        logging.info("Training started")
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
            logging.info(f"Loaded data for category: {category}")

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.9
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_tfidf, y_train
        )
        base_learners = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    random_state=42,
                    class_weight="balanced",
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
        logging.info("Model training completed")
        logging.info("Saving trained model and vectorizer")

        with open("app/trained_models/stacked_model.pkl", "wb") as model_file:
            pickle.dump(stacked_model, model_file)
        with open("app/trained_models/vectorizer.pkl", "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

        logging.info("Model and vectorizer saved successfully")
        predictions = stacked_model.predict(X_test_tfidf)
        logging.info("Model prediction completed")
        report = classification_report(y_test, predictions, zero_division=0)
        st.write(report)
        st.write(confusion_matrix(y_test, predictions))

    # Descriptive Analysis Function
    def descriptive_analysis(text):
        word_count = len(text.split())
        unique_words = len(set(text.split()))
        return word_count, unique_words

    # Generate Word Cloud
    def generate_wordcloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        return fig

    # Decision Support Functionality
    def provide_recommendations(prediction, top_skills):
        recommendations = {
            "Backend Developer": "Focus on improving your skills in Microservices and Docker.",
            "Cloud Engineer": "Consider gaining more experience with AWS and Terraform.",
            "Data Scientist": "Enhance your knowledge in Machine Learning and Data Visualization.",
            "Frontend Developer": "Strengthen your skills in React and CSS.",
            "Full Stack Developer": "Work on improving both your front-end and back-end skills.",
            "Machine Learning Engineer": "Gain more experience in Deep Learning and Model Deployment.",
            "Mobile App Developer (iOS/Android)": "Consider learning more about Flutter and Firebase.",
            "Python Developer": "Improve your skills in Django and Flask.",
        }
        return recommendations.get(prediction, "No specific recommendations available.")

    # Pie Chart for Skill Distribution
    def skill_distribution_pie_chart(skills_df):
        skills_count = skills_df["Job Category"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            skills_count, labels=skills_count.index, autopct="%1.1f%%", startangle=90
        )
        ax.axis("equal")
        return fig

    # Upload Resume Section
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
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
        result = process_resume(filepath)
        preprocessed_text = result["preprocessed_text"]
        prediction = predict_job_category(preprocessed_text)
        top_skills = extract_top_skills(preprocessed_text, prediction)

        # Descriptive analysis
        word_count, unique_words = descriptive_analysis(preprocessed_text)
        st.subheader("Descriptive Analysis")
        st.write(f"Word Count: {word_count}")
        st.write(f"Unique Words: {unique_words}")

        # Word Cloud from skills found in the resume
        st.subheader("Word Cloud")
        if top_skills:
            skills_text = " ".join([skill for skill in top_skills])
            wordcloud_plot = generate_wordcloud(skills_text)
            st.pyplot(wordcloud_plot)
        else:
            st.write("No relevant skills found to generate word cloud.")

        # Display the prediction
        st.subheader(f"Predicted Job Category: **{prediction}**")
        st.markdown("---")

        # Display top skills for the predicted job category
        st.subheader(f"Top Skills for **{prediction}**")
        st.markdown(", ".join(top_skills))

        # Bar chart for top skills comparison
        st.subheader("Top Skills Comparison")
        skills_data = {"Job Category": [], "Skill": []}
        for category in JOB_SKILLS.keys():
            other_top_skills = extract_top_skills(preprocessed_text, category)
            skills_data["Job Category"].extend([category] * len(other_top_skills))
            skills_data["Skill"].extend(other_top_skills)
        skills_df = pd.DataFrame(skills_data)
        skills_count = (
            skills_df.groupby(["Job Category", "Skill"])
            .size()
            .reset_index(name="count")
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(
            data=skills_count,
            y="Skill",
            x="count",
            hue="Job Category",
            dodge=True,
            palette="viridis",
            ax=ax,
        )
        ax.set_title("Top Skills Comparison Across Job Categories")
        st.pyplot(fig)

        st.subheader("Skill Distribution Across Job Categories")
        pie_chart = skill_distribution_pie_chart(skills_df)
        st.pyplot(pie_chart)

        # Display top skills for other job categories
        with st.expander("Compare with Other Job Categories"):
            for category in JOB_SKILLS.keys():
                if category != prediction:
                    other_top_skills = extract_top_skills(preprocessed_text, category)
                    if other_top_skills:
                        st.subheader(f"Top Skills for **{category}**")
                        st.markdown(", ".join(other_top_skills))
                    else:
                        st.subheader(f"No relevant skills found for **{category}**")
        # Provide recommendations based on the prediction and top skills
        st.subheader("Recommendations")
        recommendations = provide_recommendations(prediction, top_skills)
        st.write(recommendations)

        st.header("Search within the resume for a skill:")
        query = st.text_input("Enter a skill or keyword to search in the resume:")
        if query:
            st.write("Results for query:", query)
            results = [
                skill
                for skill in preprocessed_text.split()
                if query.lower() in skill.lower()
            ]
            st.write(set(results))

        # Clear output when new resume is uploaded
        if (
            st.session_state.uploaded_file
            and uploaded_file != st.session_state.uploaded_file
        ):
            st.session_state.uploaded_file = uploaded_file
            st.experimental_rerun()

    # Train Model Section
    st.sidebar.header("Admin Section")
    if username == "admin":
        if st.sidebar.button("Train Model"):
            train_stacked_classifier()
            st.success("Model trained successfully!")
        else:
            st.sidebar.write("Training is available for admins only.")
else:
    st.warning("Please login to access the application.")

set_sidebar()  # Set the sidebar content
