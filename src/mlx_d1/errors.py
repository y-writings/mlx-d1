"""Errors the HTTP layer turns into JSON."""


class RequestError(Exception):
    """A request failed before or during a decision, with an HTTP status."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class QuestionError(ValueError):
    """A question, criterion, or state failed validation."""


class ImageError(ValueError):
    """An image could not be read under the server's rules."""


class ConfigError(ValueError):
    """The config file or environment could not be loaded."""
