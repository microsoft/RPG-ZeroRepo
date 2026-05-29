"""Utility helper functions."""

import re


def validate_email(email: str) -> bool:
    """Validate an email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def normalize_name(name: str) -> str:
    """Normalize a user name by stripping and title-casing."""
    return name.strip().title()


def format_user_display(name: str, email: str) -> str:
    """Format a user for display."""
    return f"{normalize_name(name)} <{email}>"
