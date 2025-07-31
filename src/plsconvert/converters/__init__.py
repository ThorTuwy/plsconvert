"""
Converters package for plsconvert.

This package contains all the converter implementations and the registry system.
"""

# Import all converter modules to ensure they are registered
# ruff: noqa: F401
from . import abstract  
from . import registry
from . import universal
from .addons import audio, docs, ai, braille, media, compression, configs, threed

# Export the main classes
from .abstract import Converter
from .registry import ConverterRegistry, registerConverter
from .universal import universalConverter

__all__ = [
    "Converter",
    "ConverterRegistry", 
    "registerConverter",
    "universalConverter"
] 