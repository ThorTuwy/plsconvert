"""
Converters package for plsconvert.

This package contains all the converter implementations and the registry system.
"""

# Import all converter modules to ensure they are registered
# ruff: noqa: F401
from . import abstract  
from . import registry
from . import audio
from . import docs
from . import media
from . import compression
from . import ai
from . import braille
from . import configs
from . import universal

# Export the main classes
from .abstract import Converter
from .registry import ConverterRegistry, register_converter
from .universal import universalConverter

__all__ = [
    "Converter",
    "ConverterRegistry", 
    "register_converter",
    "universalConverter"
] 