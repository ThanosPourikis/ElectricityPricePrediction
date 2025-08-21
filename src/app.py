"""
Refactored main application entry point.

This file now uses the new Flask application factory pattern
with proper separation of concerns.
"""

from configs import config
from web.app import create_app

# Create the Flask application
app = create_app()

if __name__ == "__main__":
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)
