import os
import importlib

__all__ = []

lc_dir = os.path.join(os.path.dirname(__file__), "leetcode")
for filename in os.listdir(lc_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f"leetcode.{module_name}", package=__name__)
        globals()[module_name] = module
        __all__.append(module_name)
