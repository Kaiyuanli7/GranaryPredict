from importlib import import_module
import sys

# Import the existing granarypredict package and expose it under the new name
_base = import_module("granarypredict")
sys.modules[__name__] = _base

# Also expose submodules so `import siloflow.foo` works
for _sub in [
    "config",
    "ingestion",
    "cleaning",
    "features",
    "model",
    "evaluate",
    "utils",
]:
    try:
        sys.modules[f"{__name__}.{_sub}"] = import_module(f"granarypredict.{_sub}")
    except ModuleNotFoundError:
        # optional submodule may not exist
        pass 