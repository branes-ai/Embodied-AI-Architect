"""Exceptions for the spec system."""


class SpecError(Exception):
    """Base exception for spec operations."""


class SpecNotFoundError(SpecError):
    """Raised when a spec is not found."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Spec not found: {name}")


class SpecAlreadyExistsError(SpecError):
    """Raised when creating a spec that already exists."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Spec already exists: {name}")


class InvalidPathError(SpecError):
    """Raised when a JSON pointer path is invalid."""

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        msg = f"Invalid path: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class VersionNotFoundError(SpecError):
    """Raised when a spec version is not found."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        super().__init__(f"Version '{version}' not found for spec '{name}'")
