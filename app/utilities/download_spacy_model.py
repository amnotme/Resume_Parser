import subprocess
import spacy


def download_and_load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp
