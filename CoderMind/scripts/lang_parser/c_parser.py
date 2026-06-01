from __future__ import annotations

from ._c_family_parser import CFamilyParser
from .config.c import C_CONFIG


class CParser(CFamilyParser):
    language = "c"

    def __init__(self) -> None:
        super().__init__(C_CONFIG)
