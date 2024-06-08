import spacy
import os
from os import getenv
from app.utilities import clean_text, extract_text_from_pdf


nlp = spacy.load("en_core_web_trf")

data_dir = getenv("RESUME_DATA_FOLDER", "tmp1")
output_dir_train = getenv("TRAINED_DATA_FOLDER", "tmp2")
output_dir_parsed = getenv("PARSED_PDFS_FOLDER", "tmp3")


def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def save_text_to_file(job_type, text, output_dir, count):
    filename = f"{job_type}-{count}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(text)


def process_resumes(data_dir, output_dir_train, output_dir_parsed):
    count_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".pdf"):
                job_type = os.path.basename(root)
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                cleaned_text = clean_text(text)
                preprocessed_text = preprocess_text(cleaned_text)

                # Increment count for job type
                if job_type not in count_dict:
                    count_dict[job_type] = 0
                count_dict[job_type] += 1

                save_text_to_file(
                    job_type, preprocessed_text, output_dir_train, count_dict[job_type]
                )
                print(f"processed {job_type}-{count_dict[job_type]}!")
