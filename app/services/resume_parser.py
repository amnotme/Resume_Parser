import spacy
from os import getenv
from app.utilities.utils import (
    clean_text,
    extract_text_from_pdf,
    save_text_to_file
)
from spacy.pipeline import EntityRuler
from app.dataset import JOB_SKILLS

data_dir = getenv("RESUME_DATA_FOLDER", "tmp1")
output_dir_train = getenv("TRAINED_DATA_FOLDER", "tmp2")
output_dir_parsed = getenv("PARSED_PDFS_FOLDER", "tmp3")
output_dir_uploaded = getenv("UPLOAD_FOLDER", "tmp4")


def _load_custom_ner():
	nlp = spacy.load("en_core_web_sm")
	ruler = nlp.add_pipe("entity_ruler", before="ner")

	# Debugging: Print the patterns to verify
	patterns = [{"label": "SKILL", "pattern": skill.lower()} for skills in JOB_SKILLS.values() for skill in skills]
	for pattern in patterns[:5]:  # Print first 5 patterns for verification
		print(f"Adding pattern: {pattern}")

	ruler.add_patterns(patterns)

	return nlp


def preprocess_text(text):
    text = text.lower()
    nlp = _load_custom_ner()

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def process_resume(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)

    # Extract entities using the custom NER pipeline
    nlp = _load_custom_ner()
    doc = nlp(preprocessed_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Debugging: Print the preprocessed text and entities found
    print(f"Preprocessed text: {preprocessed_text[:200]}...")  # Print first 200 characters
    print(f"Entities found: {entities}")

    entities_text = " ".join([f"{ent[0]}_{ent[1]}" for ent in entities])

    # Combine preprocessed text with entities
    combined_text = preprocessed_text + " " + entities_text

    save_text_to_file(
        text=combined_text,
        output_dir=output_dir_parsed,
    )
    print({"entities": entities})
    return {"preprocessed_text": combined_text}