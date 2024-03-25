import importlib

working_dir = "/Users/lid/Downloads/src/ml/address-parser"
module_name = "src/serve/service.py:svc"
module = importlib.import_module(module_name, package=working_dir)


