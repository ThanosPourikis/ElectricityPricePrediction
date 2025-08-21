"""Flask application factory."""

import logging

from flask import Flask, jsonify

from configs import config
from core.exceptions import APIError, DatabaseError, ModelError, ValidationError


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_url_path="",
        static_folder="../static",
        template_folder="../templates",
    )

    # Configure logging
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", filename=config.LOG_FILE)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    from .routes import register_blueprints

    register_blueprints(app)

    return app


def register_error_handlers(app: Flask):
    """Register error handlers for the application."""

    @app.errorhandler(APIError)
    def handle_api_error(error):
        return jsonify({"error": error.message}), error.status_code

    @app.errorhandler(DatabaseError)
    def handle_database_error(error):
        app.logger.error(f"Database error: {error.message}")
        return jsonify({"error": "Database operation failed"}), error.status_code

    @app.errorhandler(ModelError)
    def handle_model_error(error):
        return jsonify({"error": f"Model error: {error.message}"}), error.status_code

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        return jsonify({"error": f"Validation error: {error.message}"}), error.status_code

    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500)
    def handle_internal_error(error):
        app.logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500
