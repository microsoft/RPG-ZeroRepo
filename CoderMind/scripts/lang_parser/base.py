from __future__ import annotations

from abc import ABC, abstractmethod

from .models import LPFileResult


class BaseLanguageParser(ABC):
    @abstractmethod
    def parse_file(self, path: str, source: str) -> LPFileResult:
        raise NotImplementedError

    @abstractmethod
    def validate_syntax(self, path: str, source: str) -> tuple[bool, str | None]:
        raise NotImplementedError
