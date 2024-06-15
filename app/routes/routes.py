import os
from os import getenv
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
from app.services import process_resumes, process_resume
from app.services.model_training import (
    train_stacked_classifier,
)
from app.services.prediction_utils import predict_job_category


resume_bp = Blueprint("resume_bp", __name__)

data_dir = getenv("RESUME_DATA_FOLDER", "tmp1")
output_dir_train = getenv("TRAINED_DATA_FOLDER", "tmp2")
output_dir_parsed = getenv("PARSED_PDFS_FOLDER", "tmp3")


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
        preprocessed_text, sections = process_resume(filepath)
        prediction = predict_job_category(preprocessed_text)
        return jsonify({"message": prediction, "sections": sections}), 201
    else:
        return jsonify({"error": "File not allowed"}), 400


@resume_bp.route("/process", methods=["POST"])
def process_data():
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_parsed, exist_ok=True)

    process_resumes(data_dir, output_dir_train)


@resume_bp.route("/train", methods=["POST"])
def train_data():
    report = train_stacked_classifier(print_predictions=True, limit_run=False)
    return jsonify({"message": "Model trained successfully!", "report": report})
