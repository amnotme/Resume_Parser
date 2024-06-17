import re
import PyPDF2
import spacy

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



def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

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
        r"(Summary|Experience|Skills|Skill|Achievements|Highlights|Awards|Projects|Recommendations)\s*[:\-]?\s*(.*?)(?=\n*(Summary|Experience|Skills|Skill|Achievements|Highlights|Awards|Projects|Recommendations)\s*[:\-]?\s*|$)",
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
