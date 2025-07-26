from typing import Callable
from plsconvert.converters.abstract import Converter
from plsconvert.utils.graph import PairList, Graph

class ConverterRegistry:
    """Centralized registry for all converters in the system."""
    
    practicalGraph: Graph = Graph()
    theoreticalGraph: Graph = Graph()

    @classmethod
    def register(cls):
        # Decorator for easy registration
        def decorator(converterClass: type[Converter]) -> type[Converter]:
            """Register all the conversions of a converter class in the registry and index its supported pairs."""

            converter = converterClass()

            # Only add the converter to the graph if its dependencies are met
            if converter.dependencies.check:
                cls.practicalGraph += converter.localGraph
            cls.theoreticalGraph += converter.localGraph

            return converterClass

        return decorator
    
def registerConverter(converterClass: type[Converter]) -> type[Converter]:
    """Decorator to register a converter and add its local graph to the registry."""
    return ConverterRegistry.register()(converterClass)
    
def addMethodData(supportedPairs: PairList, hasProgressBar: bool, *args, **kwargs) -> Callable:
    """Decorator to set the supportedPairs attribute of a function and add the hasProgressBar attribute."""
    def decorator(f: Callable) -> Callable:
        setattr(f, "supportedPairs", supportedPairs)
        setattr(f, "hasProgressBar", hasProgressBar)
        for arg in args:
            setattr(f, arg, kwargs[arg])
        for key, value in kwargs.items():
            setattr(f, key, value)
        return f
    return decorator


