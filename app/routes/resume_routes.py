from flask import Blueprint, request, jsonify
from app.services.resume_parser import parse_resume

resume_bp = Blueprint('resume_bp', __name__)
hello_bp = Blueprint('hello_bp', __name__)
@resume_bp.route('/parse_resume', methods=['POST'])
def handle_parse_resume():
    file = request.files['resume']
    result = parse_resume(file)
    return jsonify(result)


@hello_bp.route('/hello', methods=['GET'])
def hello():
    return (200, {"message": "hello"})
