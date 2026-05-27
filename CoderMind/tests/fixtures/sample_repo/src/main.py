"""Main entry point for the sample application."""

from src.models.user import User
from src.utils.helpers import validate_email


def create_user(name: str, email: str) -> User:
    """Create a new user with validated email."""
    if not validate_email(email):
        raise ValueError(f"Invalid email: {email}")
    return User(name=name, email=email)


def list_users(users: list) -> list:
    """Return sorted list of user names."""
    return sorted(u.name for u in users)


def main():
    """Application main function."""
    user = create_user("Alice", "alice@example.com")
    print(f"Created user: {user.name}")
