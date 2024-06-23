import subprocess
import spacy
import os


def download_and_load_spacy_model():
    model_name = "en_core_web_sm"
    try:
        # Check if the model is already installed
        nlp = spacy.load(model_name)
    except OSError:
        # If the model is not found, download it
        print(f"Downloading {model_name} model...")
        subprocess.run(
            [os.sys.executable, "-m", "spacy", "download", model_name], check=True
        )
        nlp = spacy.load(model_name)
    return nlp
