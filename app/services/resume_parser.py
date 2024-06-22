import spacy
import os
from os import getenv
from app.utilities import clean_text, extract_text_from_pdf, extract_sections
from app.utilities import extract_entities, save_text_to_file
import uuid
from logging import Logger

logger = Logger("resume_parser")
nlp = spacy.load("en_core_web_trf")

data_dir = getenv("RESUME_DATA_FOLDER", "tmp1")
output_dir_train = getenv("TRAINED_DATA_FOLDER", "tmp2")
output_dir_parsed = getenv("PARSED_PDFS_FOLDER", "tmp3")
output_dir_uploaded = getenv("UPLOAD_FOLDER", "tmp4")


def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def process_resumes(data_dir, output_dir_train):
    count_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".pdf"):
                job_type = os.path.basename(root)
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                cleaned_text = clean_text(text)
                if cleaned_text:
                    # extracted_sections = extract_sections(cleaned_text)
                    # extracted_sections_text = " ".join(extracted_sections.values())
                    preprocessed_text = preprocess_text(cleaned_text)
                    combined_text = preprocessed_text

                    if job_type not in count_dict:
                        count_dict[job_type] = 0
                    count_dict[job_type] += 1

                    save_text_to_file(
                        text=combined_text,
                        output_dir=output_dir_train,
                        job_type=job_type,
                        count=count_dict[job_type],
                    )
                    print(f"Processed {job_type}-{count_dict[job_type]}!")
                    # logger.info(msg=f"Processed {job_type}-{count_dict[job_type]}!")


def process_resume(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)

    # Extract entities
    entities = extract_entities(preprocessed_text)
    entities_text = " ".join([f"{ent[0]}_{ent[1]}" for ent in entities])

    # Combine preprocessed text with entities
    combined_text = preprocessed_text + " " + entities_text

    logger.info(msg=f"Saving file to /{output_dir_parsed}")
    save_text_to_file(
        text=combined_text,
        output_dir=output_dir_parsed,
    )
    logger.info(msg=f"Uploaded resume has been parsed!")
    print({"entities": entities})
    return {"preprocessed_text": combined_text}
