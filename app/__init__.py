from flask import Flask
from app.routes.routes import resume_bp


def create_app():
    app = Flask(__name__)
    app.config["ALLOWED_EXTENSIONS"] = {"pdf", "txt"}

    app.register_blueprint(resume_bp, url_prefix="/resume")

    return app
