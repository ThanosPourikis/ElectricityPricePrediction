"""Routes package."""

from .api_routes import api_bp
from .main_routes import main_bp


def register_blueprints(app):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/Api")
