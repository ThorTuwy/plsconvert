from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Any
from plsconvert.utils.dependency import Dependencies
from tqdm import tqdm
from functools import wraps

if TYPE_CHECKING:
    from plsconvert.utils.graph import ConversionAdj

class ProgressBar:
    def __init__(self, total: int, bar_format: str = '|{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]'):
        self.total = total
        self.bar_format = bar_format
        self.pbar = tqdm(total=total, bar_format=bar_format)

    def update(self, n: int = 1):
        self.pbar.update(n)

    def extend(self, n: int):
        self.pbar.total += n

    def close(self):
        self.pbar.close()

    class Registry:
        """Registry for tracking which functions have progress bar support."""
        
        def __init__(self):
            self._functions_with_progress = {}
        
        def register(self, func: Callable[..., Any], pairList: list[tuple[str, str]] | None = None) -> Callable[..., Any]:
            """Decorator to register a function as having progress bar support."""
            if pairList is None:
                pairList = []
            # Store by function name for easier lookup
            self._functions_with_progress[func.__name__] = pairList
            return func
        
        def hasProgressBar(self, func_name: str, pair: tuple[str, str]) -> bool:
            """Check if a function has progress bar support for a specific pair."""
            if func_name not in self._functions_with_progress:
                return False
            return pair in self._functions_with_progress[func_name]
        
        def getFunctionsWithProgressBar(self) -> dict[str, list[tuple[str, str]]]:
            """Get all functions with progress bar support."""
            return self._functions_with_progress.copy()
        
        def getPairsForFunction(self, func_name: str) -> list[tuple[str, str]]:
            """Get all pairs supported by a specific function."""
            return self._functions_with_progress.get(func_name, [])

# Global registry instance
pbRegistry = ProgressBar.Registry()

def withProgressBar(pairList: list[tuple[str, str]] | None = None):
    """Decorator to mark a function as having progress bar support."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return pbRegistry.register(func, pairList or [])
    return decorator

class Converter(ABC):
    """
    Abstract class for all converters.
    """

    def __init__(self):
        self.description = self.__class__.__doc__ or "No description available"
        self.progressBar = None

    def exist(self) -> bool:
        return True

    def adj(self) -> "ConversionAdj":
        from plsconvert.utils.graph import ConversionAdj, Conversion
        old_adj = self.adjConverter()
        adj = {}
        for source in old_adj:
            new_targets = []
            for target_tuple in old_adj[source]:
                # target_tuple is (target_format, source_format) from conversionFromToAdj
                target_format = target_tuple[0]
                if target_format not in adj:
                    adj[target_format] = []
                if source != target_format:  # Avoid self-loops
                    new_targets.append(Conversion((target_format, self)))
            adj[source] = new_targets
        return ConversionAdj(adj)
    
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
        return Dependencies.empty()

    def hasProgressBar(self, function: Callable[..., Any], pair: tuple[str, str]) -> bool:
        """
        Check if a specific function has progress bar support for a specific pair.
        
        Args:
            function: Function to check
            pair: Pair of input and output extensions
            
        Returns:
            True if the function supports progress bars for this pair, False otherwise
        """
        return pbRegistry.hasProgressBar(function.__name__, pair)

    def hasProgressBar4Pair(self, pair: tuple[str, str]) -> bool:
        """
        Check if this converter has progress bar support for a specific conversion.
        
        Args:
            pair: Pair of input and output extensions
            
        Returns:
            True if the conversion supports progress bars, False otherwise
        """
        # Check if any function in this converter supports this pair
        for func_name in self.getFunctionsWithProgressBar():
            if pbRegistry.hasProgressBar(func_name, pair):
                return True
        return False

    def getFunctionsWithProgressBar(self) -> set[str]:
        """Get all functions with progress bar support for this converter."""
        # For now, return all registered functions - filtering by instance can be added later
        return set(pbRegistry.getFunctionsWithProgressBar().keys())

    @abstractmethod
    def adjConverter(self) -> "ConversionAdj":
        pass

    @abstractmethod
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        pass

    def pbInit(self, total: int, bar_format: str = '|{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]') -> ProgressBar:
        self.progressBar = ProgressBar(total=total, bar_format=bar_format)
        return self.progressBar
