from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from plsconvert.utils.dependency import Dependencies

if TYPE_CHECKING:
    from plsconvert.utils.graph import ConversionAdj

class Converter(ABC):
    """
    Abstract class for all converters.
    """

    def __init__(self):
        self.description = self.__class__.__doc__ or "No description available"

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

    @abstractmethod
    def adjConverter(self) -> "ConversionAdj":
        pass

    @abstractmethod
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        pass
    