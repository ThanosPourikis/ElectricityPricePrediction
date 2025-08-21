"""Web package for Flask application."""

from .app import create_app
from .routes import register_blueprints

__all__ = ["create_app", "register_blueprints"]
