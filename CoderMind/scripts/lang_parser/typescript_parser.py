from __future__ import annotations

from ._ecmascript_parser import ECMAScriptParser
from .config.typescript import TYPESCRIPT_CONFIG


class TypeScriptParser(ECMAScriptParser):
    language = "typescript"

    def __init__(self) -> None:
        super().__init__(TYPESCRIPT_CONFIG)
