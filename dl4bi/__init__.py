import importlib
import os
import sys

# re-export all dl4bi.core.<module> as dl4bi.<module>
core_path = os.path.join(os.path.dirname(__file__), "core")
for module in os.listdir(core_path):
    if module.endswith(".py") and module != "__init__.py":
        module_name = module[:-3]  # Remove `.py` extension
        full_module_name = f"dl4bi.core.{module_name}"
        imported_module = importlib.import_module(full_module_name)
        sys.modules[f"dl4bi.{module_name}"] = imported_module
        globals()[module_name] = imported_module
