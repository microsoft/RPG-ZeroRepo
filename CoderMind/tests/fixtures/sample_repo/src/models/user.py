"""User model definition."""


class User:
    """Represents a user in the system."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self._active = True

    def deactivate(self):
        """Mark the user as inactive."""
        self._active = False

    def is_active(self) -> bool:
        """Check if the user is currently active."""
        return self._active

    def to_dict(self) -> dict:
        """Serialize user to dictionary."""
        return {
            "name": self.name,
            "email": self.email,
            "active": self._active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Deserialize user from dictionary."""
        user = cls(name=data["name"], email=data["email"])
        user._active = data.get("active", True)
        return user
