#!/usr/bin/env python3
"""
Deprecated entry point. Use src/create_quantized_mobilenet.py instead.

This file is a thin forwarding shim kept for backward compatibility with
existing scripts and documentation that still invoke the v3-only name.
The original v3 behavior is preserved via the default --backbone v3.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so the import below works regardless of CWD
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.create_quantized_mobilenet import main

print(
    "[deprecated] create_quantized_mobilenet_v3.py is deprecated; "
    "use create_quantized_mobilenet.py "
    "(the v3 behavior is preserved as --backbone v3, the default).",
    file=sys.stderr,
)

if __name__ == '__main__':
    sys.exit(main())
