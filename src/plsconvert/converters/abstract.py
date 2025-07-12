from abc import ABC, abstractmethod
from pathlib import Path


class Converter(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    def exist(self) -> bool:
        return True

    def adj(self) -> dict[str, list[list[str]]]:
        adj = self.adjConverter()
        for key in adj:
            adj[key] = [[ext, self.name] for ext in adj[key]]
        return adj

    @abstractmethod
    def adjConverter(self) -> dict[str, list[list[str]]]:
        pass

    @abstractmethod
    def convert(
        self, input: Path, output: Path, input_extension: str, output_extension: str
    ) -> None:
        pass

    @abstractmethod
    def metDependencies(self) -> bool:
        pass
