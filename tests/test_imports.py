import importlib
import pkgutil

import echora_kara


def test_every_module_imports():
    for module in pkgutil.walk_packages(echora_kara.__path__, "echora_kara."):
        importlib.import_module(module.name)
