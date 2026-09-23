"""Absolute-path subprocess bootstrap; never import the CLI from the workspace."""

from pathlib import Path
import sys


if __name__ == "__main__":
    # -I ignores PYTHONIOENCODING/PYTHONUTF8. Preserve the CLI's UTF-8 contract
    # for redirected Windows output without trusting interpreter environment.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cmind_cli import main

    main()
