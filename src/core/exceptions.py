"""Custom exceptions for the application."""


class APIError(Exception):
    """Base exception for API-related errors."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DatabaseError(APIError):
    """Exception for database-related errors."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)


class ModelError(APIError):
    """Exception for model-related errors."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message, status_code)


class ValidationError(APIError):
    """Exception for validation errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)
