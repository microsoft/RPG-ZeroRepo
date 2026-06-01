from __future__ import annotations

from ._ecmascript_parser import ECMAScriptParser
from .config.javascript import JAVASCRIPT_CONFIG


class JavaScriptParser(ECMAScriptParser):
    language = "javascript"

    def __init__(self) -> None:
        super().__init__(JAVASCRIPT_CONFIG)
