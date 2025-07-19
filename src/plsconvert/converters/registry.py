from typing import Dict, List, Type
from plsconvert.converters.abstract import Converter

class ConverterRegistry:
    """Centralized registry for all converters in the system."""
    
    _converters: Dict[str, Type[Converter]] = {}
    
    @classmethod
    def register(cls, converter_class: Type[Converter]) -> Type[Converter]:
        """Register a converter class in the registry."""
        cls._converters[converter_class.__name__] = converter_class
        return converter_class
    
    @classmethod
    def get_all_converters(cls) -> List[Type[Converter]]:
        """Get all registered converter classes."""
        return list(cls._converters.values())
    
    @classmethod
    def get_converter_by_name(cls, name: str) -> Type[Converter]:
        """Get a specific converter by name."""
        if name not in cls._converters:
            raise KeyError(f"Converter '{name}' not found in registry")
        return cls._converters[name]
    
    @classmethod
    def get_converter_names(cls) -> List[str]:
        """Get all registered converter names."""
        return list(cls._converters.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered converters (mainly for testing)."""
        cls._converters.clear()

# Decorator for easy registration
def register_converter(converter_class: Type[Converter]) -> Type[Converter]:
    """Decorator to automatically register a converter class."""
    return ConverterRegistry.register(converter_class) 