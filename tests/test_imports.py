"""Smoke test: every module of the package imports."""

import importlib
import pkgutil

import echora_kara


def test_every_module_imports():
    """Every module under echora_kara imports without error."""
    for module in pkgutil.walk_packages(echora_kara.__path__, "echora_kara."):
        importlib.import_module(module.name)
