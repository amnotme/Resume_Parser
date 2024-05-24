import os
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
from app.services.resume_parser import (
    extract_text_from_pdf,
    extract_sections,
    clean_text,
    preprocess_text,
    process_resumes,
)

resume_bp = Blueprint("resume_bp", __name__)

data_dir = "resume_parser_data"
output_dir_train = "train_texts"
output_dir_parsed = "parsed_pdfs"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf", "txt"}


@resume_bp.route("/")
def index():
    return render_template("upload.html")


@resume_bp.route("/upload", methods=["POST"])
def handle_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(os.getenv("UPLOAD_FOLDER", "/tmp"), filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)
        cleaned_text = clean_text(text)
        preprocessed_text = preprocess_text(cleaned_text)
        sections = extract_sections(preprocessed_text)
        # sections = extract_sections(cleaned_text)
        return jsonify(sections)
    else:
        return jsonify({"error": "File not allowed"}), 400


@resume_bp.route("/process", methods=["POST"])
def process_data():
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_parsed, exist_ok=True)

    process_resumes(data_dir, output_dir_train, output_dir_parsed)
