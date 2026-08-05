from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    source = str(source_root)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)