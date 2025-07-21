import os
import importlib

__all__ = []

problems_dir = os.path.join(os.path.dirname(__file__), "problems")
for filename in os.listdir(problems_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f"problems.{module_name}", package=__name__)
        globals()[module_name] = module
        __all__.append(module_name)
