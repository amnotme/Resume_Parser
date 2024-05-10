from flask import Flask

def create_app():
    app = Flask(__name__)
    # Additional configuration can be loaded here

    from .routes.resume_routes import resume_bp
    app.register_blueprint(resume_bp)

    return app
