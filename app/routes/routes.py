import os
from os import getenv
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
from app.services import process_resume
from app.services.model_training import (
    train_stacked_classifier,
)
from app.services.prediction_utils import predict_job_category
from app.utilities import extract_top_skills

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

        # Preprocess the resume
        preprocessed_text = process_resume(filepath)["preprocessed_text"]

        # Predict the job category
        prediction = predict_job_category(preprocessed_text)

        # Extract top skills
        top_skills = extract_top_skills(preprocessed_text, prediction)
        print({"message": prediction, "top_skills": top_skills})
        return jsonify({"message": prediction, "top_skills": top_skills}), 201
    else:
        return jsonify({"error": "File not allowed"}), 400


@resume_bp.route("/train", methods=["POST"])
def train_data():
    report = train_stacked_classifier(print_predictions=True)
    return jsonify({"message": "Model trained successfully!", "report": report})
