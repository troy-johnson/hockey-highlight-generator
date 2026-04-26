# tests/conftest.py
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "v2", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "v3", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "v3", "resolve_scripts"))
