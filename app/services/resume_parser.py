import PyPDF2
import re
import spacy
import os

nlp = spacy.load("en_core_web_sm")

data_dir = "resume_parser_data"
output_dir_train = "train_texts"
output_dir_parsed = "parsed_pdfs"


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


def extract_sections(text):
    sections = {}
    # Updated regex pattern to be more robust and flexible
    pattern = re.compile(
        r"(Summary|Highlights|Accomplishments|Experience|Education|Skills)\s*[:\-]?\s*(.*?)(?=\n*(Summary|Highlights|Accomplishments|Experience|Education|Skills)\s*[:\-]?\s*|$)",
        re.I | re.DOTALL,
    )
    matches = pattern.finditer(text)
    for match in matches:
        section_title = match.group(1).lower()  # Normalize the key to lowercase
        if section_title in sections:
            # Append new content with a newline for readability if the section already exists
            sections[section_title] += match.group(2).strip()
        else:
            sections[section_title] = match.group(2).strip()
    return sections


def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def save_text(job_type, text, output_dir, count):
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

                # Save text
                save_text(
                    job_type, preprocessed_text, output_dir_train, count_dict[job_type]
                )
                print(f"processed {job_type}-{count_dict[job_type]}!")
                # save_text(
                #     job_type, cleaned_text, output_dir_parsed, count_dict[job_type]
                # )
