from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

from .base import BaseLanguageParser
from .config import C_CONFIG, CPP_CONFIG, GO_CONFIG, JAVASCRIPT_CONFIG, PYTHON_CONFIG, RUST_CONFIG, TYPESCRIPT_CONFIG
from .models import LanguageConfig, LPFileResult, NotSupported

_CONFIGS: dict[str, LanguageConfig] = {
    PYTHON_CONFIG.name: PYTHON_CONFIG,
    GO_CONFIG.name: GO_CONFIG,
    TYPESCRIPT_CONFIG.name: TYPESCRIPT_CONFIG,
    JAVASCRIPT_CONFIG.name: JAVASCRIPT_CONFIG,
    C_CONFIG.name: C_CONFIG,
    CPP_CONFIG.name: CPP_CONFIG,
    RUST_CONFIG.name: RUST_CONFIG,
}
_PARSERS: dict[str, BaseLanguageParser] = {}


def _normalize_path(path: str) -> str:
    file_part = str(path).split(":", 1)[0]
    return PurePosixPath(file_part.replace("\\", "/")).as_posix().removeprefix("./")


def detect_language(path: str) -> str | None:
    normalized = _normalize_path(path).lower()
    for config in _CONFIGS.values():
        if any(normalized.endswith(extension) for extension in config.extensions):
            return config.name
    return None


def is_supported_source(path: str) -> bool:
    return detect_language(path) in _CONFIGS


def is_test_file(path: str) -> bool:
    config = get_config_for_path(path)
    if config is None:
        return False
    normalized = _normalize_path(path).lower()
    return any(fnmatch.fnmatchcase(normalized, pattern.lower()) for pattern in config.test_globs)


def get_config(language: str) -> LanguageConfig:
    key = language.lower()
    try:
        return _CONFIGS[key]
    except KeyError as exc:
        raise NotSupported(f"Unsupported language: {language}") from exc


def get_config_for_path(path: str) -> LanguageConfig | None:
    language = detect_language(path)
    if language is None:
        return None
    return get_config(language)


def get_parser(language: str) -> BaseLanguageParser:
    key = language.lower()
    if key not in _CONFIGS:
        raise NotSupported(f"Unsupported language: {language}")
    if key not in _PARSERS:
        if key == "python":
            from .python_parser import PythonParser

            _PARSERS[key] = PythonParser()
        elif key == "go":
            from .go_parser import GoParser

            _PARSERS[key] = GoParser()
        elif key == "typescript":
            from .typescript_parser import TypeScriptParser

            _PARSERS[key] = TypeScriptParser()
        elif key == "javascript":
            from .javascript_parser import JavaScriptParser

            _PARSERS[key] = JavaScriptParser()
        elif key == "c":
            from .c_parser import CParser

            _PARSERS[key] = CParser()
        elif key == "cpp":
            from .cpp_parser import CppParser

            _PARSERS[key] = CppParser()
        elif key == "rust":
            from .rust_parser import RustParser

            _PARSERS[key] = RustParser()
        else:
            raise NotSupported(f"Unsupported language: {language}")
    return _PARSERS[key]


def get_parser_for_file(path: str) -> BaseLanguageParser | None:
    language = detect_language(path)
    if language is None:
        return None
    return get_parser(language)


def parse_file(path: str, source: str) -> LPFileResult:
    parser = get_parser_for_file(path)
    if parser is None:
        raise NotSupported(f"Unsupported source file: {path}")
    return parser.parse_file(path, source)


def validate_syntax(path: str, source: str) -> tuple[bool, str | None]:
    parser = get_parser_for_file(path)
    if parser is None:
        return False, f"Unsupported source file: {path}"
    return parser.validate_syntax(path, source)


def markdown_fence_for_path(path: str) -> str:
    config = get_config_for_path(path)
    if config is None:
        return "text"
    return config.markdown_fence
