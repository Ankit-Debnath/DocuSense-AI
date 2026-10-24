import os
from flask import Flask, flash, redirect, url_for, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from flask_wtf.csrf import CSRFProtect, CSRFError
from config import Config
from routes.pdf_routes import pdf_bp
from database.db import init_app as init_db_app

csrf = CSRFProtect()

def create_app(config_class=Config):
    """
    Application factory that initializes and configures the Flask application.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required runtime directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["INSTANCE_FOLDER"], exist_ok=True)

    # Initialize SQLite database and teardown handlers
    init_db_app(app)

    # Initialize CSRF protection globally
    csrf.init_app(app)

    # Register blueprints
    app.register_blueprint(pdf_bp)

    # Handle file uploads that exceed MAX_CONTENT_LENGTH (10 MB)
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(error):
        flash("File is too large! Maximum allowed file size is 10 MB.", "error")
        return redirect(url_for("pdf.index"))

    # User-friendly error handler for expired or missing CSRF security tokens
    @app.errorhandler(CSRFError)
    def handle_csrf_error(error):
        if request.is_json:
            return jsonify({
                "success": False,
                "error": f"CSRF validation failed: {error.description}"
            }), 400
        flash(f"Security validation failed: {error.description}. Please refresh and submit the form again.", "error")
        return render_template("index.html"), 400

    return app

# Main app instance for running directly or via test clients
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
