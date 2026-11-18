"""Model plug-in discovery. Drop a file here following CONTRIBUTING section 5; no registry edits."""
import importlib
import pkgutil

REQUIRED = ("NAME", "FAMILY", "DEFAULT_PARAMS", "build")


def discover() -> dict:
    registry = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        if hasattr(mod, "NAME") and hasattr(mod, "build"):
            registry[mod.NAME] = mod
    return registry
