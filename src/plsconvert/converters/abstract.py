from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, final
from plsconvert.utils.dependency import Dependencies
from plsconvert.converters.progressbar import ProgressBar
from plsconvert.utils.graph import Pair

if TYPE_CHECKING:
    from plsconvert.utils.graph import Graph
    from plsconvert.converters.progressbar import ProgressBar

class Converter(ABC):
    """
    Abstract class for all converters.
    """

    def __init__(self):
        self.description = self.__class__.__doc__ or "No description available"
        self.progressBar = None

    def exist(self) -> bool:
        return True
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.name

    @property
    @abstractmethod
    def dependencies(self) -> Dependencies:
        """
        Get the dependencies for the converter.
        """
        return Dependencies.empty()
    
    @property
    def localGraph(self) -> "Graph":
        """
        Generate a local graph for the converter.
        """
        if not hasattr(self, "_localGraph"):
            from plsconvert.utils.graph import Graph, Conversion, ConversionData
            graph = Graph()
            for attr_name in self.__class__.__dict__:
                attr = self.__class__.__dict__[attr_name]
                # Check if the attribute is a callable and has 'supportedPairs'
                if callable(attr) and hasattr(attr, "supportedPairs"):
                    pairs = getattr(attr, "supportedPairs")
                    # Add a Conversion for each supported pair
                    for pair in pairs:
                        graph += Conversion(
                            pair=pair,
                            conversionData=ConversionData(
                                converter=self,
                                methodName=attr_name,
                                hasProgressBar=getattr(attr, "hasProgressBar", False),
                            )
                        )
            self._localGraph = graph
        return self._localGraph
    
    @final
    def convert(self, input: Path, output: Path, input_extension: str, output_extension: str, *args, **kwargs) -> None:
        """
        Dispatch conversion to the appropriate method based on input and output extensions.
        """ 
        conversions = self.localGraph[input_extension].findByOutput(output_extension)
        for conversion in conversions:
            return getattr(self, conversion.methodName)(input, output, input_extension, output_extension, *args, **kwargs)
        raise NotImplementedError(f"No conversion method found for {input_extension} -> {output_extension}")
    
    @classmethod
    def hasPairProgressBar(cls, pair: Pair) -> bool:
        """Check if any function in the converter has progress bar support for a specific pair."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and hasattr(attr, "supportedPairs"):
                pairs = getattr(attr, "supportedPairs")
                if pair in pairs:
                    for attr_name in dir(cls):
                        attr = getattr(cls, attr_name)
                        if callable(attr) and hasattr(attr, "hasProgressBar"):
                            return attr.hasProgressBar
        return False

    def pbInit(self, total: int, bar_format: str = '|{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]') -> ProgressBar:
        """
        Initialize the progress bar.
        """
        self.progressBar = ProgressBar(total=total, bar_format=bar_format)
        return self.progressBar
