"""
Module: tests.conftest
Configuration pytest : chemins d'import du package ``src``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet ``import src`` depuis la racine du dépôt sans installation editable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
