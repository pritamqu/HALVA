import sys
import importlib

def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None

def is_wandb_available() -> bool:
    return importlib.util.find_spec("wandb") is not None

def is_transformers_greater_than(version: str) -> bool:
    _transformers_version = importlib.metadata.version("transformers")
    return _transformers_version > version