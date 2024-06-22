import re
import PyPDF2
import spacy
from app.dataset import JOB_SKILLS
import uuid
import os
import re

nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    # Normalize common unicode characters
    text = text.replace("â€‹", "").replace("Â", "").replace("ï¼​", "").replace("â—​", "")
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove specific unwanted characters
    text = re.sub(r"[\r\n\t]", " ", text)
    # Replace unicode bullet points and other non-standard punctuation
    text = text.replace("\u2022", "")  # Bullet points
    text = text.replace("\uf0b7", "")  # Another type of bullet
    # Optionally remove any characters that are not standard printable characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Removes non-ASCII characters
    return text.strip()


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
    return full_text


def extract_top_skills(preprocessed_text, job_category):
    skills = JOB_SKILLS.get(job_category, [])
    preprocessed_text = preprocessed_text.lower()  # Ensure the text is lowercased
    top_skills = []

    for skill in skills:
        skill_lower = skill.lower()
        # Use regex to match whole words only
        if re.search(r"\b" + re.escape(skill_lower) + r"\b", preprocessed_text):
            top_skills.append(skill)

    return top_skills


def save_text_to_file(text, output_dir, job_type=None, count=0):
    if job_type and count:
        filename = f"{job_type}-{count}.txt"
    else:
        filename = f"{uuid.uuid4()}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(text)
