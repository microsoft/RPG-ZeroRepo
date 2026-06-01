from __future__ import annotations

from ._c_family_parser import CFamilyParser
from .config.cpp import CPP_CONFIG


class CppParser(CFamilyParser):
    language = "cpp"

    def __init__(self) -> None:
        super().__init__(CPP_CONFIG)
