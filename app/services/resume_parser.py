import PyPDF2
import re


def clean_text(text):
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
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
    pattern = re.compile(
        r"(Summary|Highlights|Accomplishments|Experience)\s*:\s*(.*?)(?=(Summary|Highlights|Accomplishments|Experience)\s*:\s*|$)",
        re.I | re.S,
    )
    matches = pattern.finditer(text)
    for match in matches:
        sections[match.group(1)] = match.group(2).strip()
    return sections
